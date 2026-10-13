"""ML1 to ML7: compress the training matrix, recommend, evaluate, compare and segment.

    python model_analysis.py            # reuse cached fits whose matrix and settings still match
    python model_analysis.py --force    # refit every model from scratch

Run python build_matrix.py first. Every fit runs in its own fresh Python subprocess
(`--worker`), so its fit time and peak memory are not contaminated by earlier fits.
The workers save factors and top-10 lists to the git-ignored data/processed/models/.
The main process then evaluates them on the held-out purchases and writes aggregate
reports (no customer IDs) to reports/ and figures to figures/.

Nothing here reads the test file until evaluation: k, the solver, the purchase value
and the segments are all fixed from training data alone (R2).
"""

import os
import tempfile

# Pin numerical libraries to one thread so timings are repeatable (set before numpy loads).
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _var in THREAD_VARS:
    os.environ.setdefault(_var, "1")
_tmp = "/private/tmp" if os.path.isdir("/private/tmp") else tempfile.gettempdir()
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_tmp, "ds5020-mpl"))

import psutil

_RSS_BEFORE_IMPORTS = psutil.Process().memory_info().rss

import argparse
import hashlib
import inspect
import json
import math
import platform
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
from scipy.stats import binomtest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize

from build_matrix import PROCESSED_DIR, REPORTS_DIR, ROOT, purchase_value

MODELS_DIR = PROCESSED_DIR / "models"
FIGURES_DIR = ROOT / "figures"

SEED = 42
CHOSEN_K = 100                       # predeclared in ML2 of the partner notebook
K_SWEEP = (10, 20, 50, 100, 200)
TOP_N = 10
MAIN_WEIGHTING = "log"
WEIGHTINGS = ("count", "binary", "log")
SOLVER = "arpack"                    # exact truncated SVD; same solver family as PCA below
N_SEGMENTS = 4
KMEANS_N_INIT = 20
TIMING_REPEATS = 5
RSS_SAMPLE_SECONDS = 0.001
BOOTSTRAP_SAMPLES = 2000
FACTOR_TARGET_MB = 10.0
NIGHTLY_TARGET_SECONDS = 60.0
MB = 1e6                             # decimal megabytes everywhere

# Every fit the analysis needs. Each runs once in a fresh worker process.
FIT_CONFIGS = (
    [("svd", k, MAIN_WEIGHTING) for k in K_SWEEP]
    + [("pca", CHOSEN_K, MAIN_WEIGHTING), ("pca", max(K_SWEEP), MAIN_WEIGHTING)]
    + [("svd", CHOSEN_K, w) for w in WEIGHTINGS if w != MAIN_WEIGHTING]
)


# ---------------------------------------------------------------- data

def load_quantity_matrix(processed_dir=PROCESSED_DIR):
    return sp.load_npz(Path(processed_dir) / "customer_product_quantity.npz").tocsr()


def weighted_matrix(Q, weighting):
    """Same sparsity pattern as the quantity matrix, with the chosen purchase value stored."""
    A = Q.astype(np.float64, copy=True)
    A.data = purchase_value(A.data, weighting)
    return A


def load_matrix(weighting=MAIN_WEIGHTING, processed_dir=PROCESSED_DIR):
    if weighting == MAIN_WEIGHTING:
        return sp.load_npz(Path(processed_dir) / "customer_product.npz").tocsr()
    return weighted_matrix(load_quantity_matrix(processed_dir), weighting)


def matrix_fingerprint(processed_dir=PROCESSED_DIR):
    """Hash of the quantity and stored log matrix contents (not the .npz files, whose zip
    timestamps change)."""
    digest = hashlib.sha256()
    for M in (load_quantity_matrix(processed_dir), load_matrix(MAIN_WEIGHTING, processed_dir)):
        digest.update(repr(M.shape).encode())
        for array in (M.indptr, M.indices, M.data):
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------- ML2 / ML5 decompositions

def fit_decomposition(A, method, k, seed=SEED):
    """Truncated SVD on A as it is, or PCA with implicit centring on the same sparse A."""
    if method == "svd":
        return TruncatedSVD(n_components=k, algorithm=SOLVER, random_state=seed).fit(A)
    if method == "pca":
        return PCA(n_components=k, svd_solver="arpack", random_state=seed).fit(A)
    raise ValueError(method)


def factor_arrays(model, A):
    """Customer factors (rows x k), product factors (products x k) and PCA's mean vector."""
    customer = model.transform(A)
    product = model.components_.T.copy()
    mean = getattr(model, "mean_", None)
    return customer, product, mean


def relative_error_svd(A, components):
    """||A - A V V^T||_F / ||A||_F without forming a dense matrix.

    V has orthonormal columns, so ||A - A V V^T||^2 = ||A||^2 - ||A V||^2.
    """
    total = A.multiply(A).sum()
    kept = np.square(A @ components.T).sum()
    return float(np.sqrt(max(total - kept, 0.0) / total))


def relative_error_pca(A, components, mean):
    """PCA rebuilds A as mean + (A - mean) W W^T. Returns that error relative to ||A||
    (comparable with the SVD number) and relative to the centred norm."""
    n = A.shape[0]
    total = A.multiply(A).sum()
    centred_total = total - n * mean @ mean
    projected = A @ components.T - mean @ components.T
    residual = max(centred_total - np.square(projected).sum(), 0.0)
    return float(np.sqrt(residual / total)), float(np.sqrt(residual / centred_total))


def factor_bytes(customer, product, mean=None):
    total = customer.nbytes + product.nbytes
    return total + (mean.nbytes if mean is not None else 0)


# ---------------------------------------------------------------- ML3 recommendations

def product_vectors(product_factors):
    """One unit-length row per product, so a dot product is cosine similarity."""
    return normalize(product_factors)


def top_n(scores, n=TOP_N):
    """Top n columns per row, highest score first, ties to the lower index.

    Excluded entries must already be -inf. The result is always n wide: rows with fewer
    than n allowed entries (including catalogues smaller than n) are padded with index -1
    and score -inf, so callers must drop negative indices before looking products up.
    """
    scores = np.atleast_2d(scores)
    kept = min(n, scores.shape[1])
    order = np.argsort(-scores, axis=1, kind="stable")[:, :kept]
    best = np.take_along_axis(scores, order, axis=1)
    idx = np.full((scores.shape[0], n), -1, dtype=np.int64)
    score = np.full((scores.shape[0], n), -np.inf, dtype=np.result_type(scores.dtype, np.float32))
    idx[:, :kept] = np.where(np.isneginf(best), -1, order)
    score[:, :kept] = best
    return idx, score


def item_neighbors(P, n=TOP_N, batch=1024):
    """Top-n most similar products for every product, in batches (no full product x product
    matrix is kept)."""
    m = P.shape[0]
    idx = np.empty((m, n), dtype=np.int32)
    score = np.empty((m, n), dtype=np.float32)
    for start in range(0, m, batch):
        stop = min(start + batch, m)
        sims = P[start:stop] @ P.T
        sims[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        idx[start:stop], score[start:stop] = top_n(sims, n)
    return idx, score


def recommend_similar(item_index, P, n=TOP_N):
    """ML3: the n products whose factor vectors are most cosine-similar to item_index."""
    sims = P @ P[item_index]
    sims[item_index] = -np.inf
    idx, score = top_n(sims, n)
    return idx[0], score[0]


def _mask_seen(scores, A_rows):
    rows, cols = A_rows.nonzero()
    scores[rows, cols] = -np.inf
    return scores


def customer_scores(A_rows, P):
    """Score = sum over bought products j of a_ij * cos(j, candidate): profile = A P, score = profile P^T."""
    return np.asarray((A_rows @ P) @ P.T)


def recommend_for_customer(row, A, P, n=TOP_N):
    """ML3: the n products most similar to what this customer bought, excluding anything bought."""
    A_row = A[row]
    scores = _mask_seen(customer_scores(A_row, P), A_row)
    idx, score = top_n(scores, n)
    return idx[0], score[0]


def recommend_all_customers(A, P, n=TOP_N, batch=1024):
    out = np.empty((A.shape[0], n), dtype=np.int32)
    for start in range(0, A.shape[0], batch):
        A_rows = A[start:start + batch]
        out[start:start + A_rows.shape[0]] = top_n(_mask_seen(customer_scores(A_rows, P), A_rows), n)[0]
    return out


def best_seller_totals(Q):
    """Training quantity summed per catalogue product (the best-seller ranking)."""
    return np.asarray(Q.sum(axis=0)).ravel()


def best_seller_recommendations(A, totals, n=TOP_N, batch=1024):
    """R3 baseline: the n best sellers each customer has not bought yet (refilled from lower ranks)."""
    out = np.empty((A.shape[0], n), dtype=np.int32)
    for start in range(0, A.shape[0], batch):
        A_rows = A[start:start + batch]
        scores = np.tile(totals.astype(np.float64), (A_rows.shape[0], 1))
        out[start:start + A_rows.shape[0]] = top_n(_mask_seen(scores, A_rows), n)[0]
    return out


# ---------------------------------------------------------------- ML4 evaluation

def heldout_targets(test, customer_ids, stock_codes):
    """Held-out (customer, product) pairs mapped onto the training matrix, with coverage counts.

    The primary cohort is every warm matrix customer with at least one held-out product that is
    in the training catalogue. Cold customers and out-of-catalogue products are counted, not hidden.
    """
    pairs = test[["Customer ID", "StockCode"]].drop_duplicates()
    customer_ids = np.asarray(customer_ids)
    row = pd.Series(np.arange(len(customer_ids)), index=customer_ids)
    col = pd.Series(np.arange(len(stock_codes)), index=pd.Index(stock_codes))
    pairs = pairs.assign(row=pairs["Customer ID"].map(row), col=pairs["StockCode"].map(col))
    warm = pairs["row"].notna()
    in_catalogue = pairs["col"].notna()
    eligible = pairs.loc[warm & in_catalogue]

    targets = {}
    for r, cols in eligible.groupby("row")["col"]:
        targets[int(r)] = np.sort(cols.astype(int).to_numpy())

    customers = pairs.groupby("Customer ID").agg(warm=("row", lambda s: s.notna().any()))
    n_customers = len(customers)
    n_warm = int(customers["warm"].sum())
    n_pairs = len(pairs)
    n_products = int(pairs["StockCode"].nunique())
    n_products_in_catalogue = int(pairs.loc[in_catalogue, "StockCode"].nunique())
    # Three different coverage rates: customers, (customer, product) pairs and distinct products.
    coverage = {
        "total_test_customers": n_customers,
        "warm_test_customers": n_warm,
        "cold_start_customers": n_customers - n_warm,
        "warm_customers_no_eligible_product": n_warm - len(targets),
        "eligible_customers": len(targets),
        "eligible_customer_coverage_pct": 100 * len(targets) / n_customers,
        "total_test_pairs": n_pairs,
        "eligible_test_pairs": len(eligible),
        "cold_customer_pairs": int((~warm).sum()),
        "warm_pairs_product_not_in_catalogue": int((warm & ~in_catalogue).sum()),
        "eligible_pair_coverage_pct": 100 * len(eligible) / n_pairs,
        "test_products": n_products,
        "test_products_in_catalogue": n_products_in_catalogue,
        "eligible_product_coverage_pct": 100 * n_products_in_catalogue / n_products,
    }
    return targets, coverage


def evaluate(recommendations, targets, coverage):
    """Hit rate at 10: a customer counts once if one or more held-out products is in their top 10.
    Recall at 10: the share of each customer's held-out products found in their top 10, averaged
    over customers (no cap on the denominator, so 1 hit out of 20 targets is 0.05)."""
    rows = np.array(sorted(targets))
    hit_counts = np.array([np.isin(targets[r], recommendations[r]).sum() for r in rows])
    n_targets = np.array([len(targets[r]) for r in rows])
    hits = int((hit_counts > 0).sum())
    return {
        "hit_rate_at_10": hits / len(rows),
        "hits": hits,
        "evaluated_customers": len(rows),
        "recall_at_10": float(np.mean(hit_counts / n_targets)),
        "warm_hit_rate_at_10": hits / coverage["warm_test_customers"],
        "all_test_hit_rate_at_10": hits / coverage["total_test_customers"],
        "per_customer_hit": (hit_counts > 0),
    }


def paired_comparison(hit_a, hit_b, n_boot=BOOTSTRAP_SAMPLES, seed=SEED):
    """Paired bootstrap 95% interval for rate(a) - rate(b) and an exact McNemar test."""
    hit_a, hit_b = np.asarray(hit_a, float), np.asarray(hit_b, float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(hit_a), size=(n_boot, len(hit_a)))
    diffs = hit_a[draws].mean(axis=1) - hit_b[draws].mean(axis=1)
    only_a = int(((hit_a == 1) & (hit_b == 0)).sum())
    only_b = int(((hit_a == 0) & (hit_b == 1)).sum())
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return {
        "difference": float(hit_a.mean() - hit_b.mean()),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "only_first_hit": only_a, "only_second_hit": only_b,
        "mcnemar_exact_p": float(p),
    }


def bootstrap_interval(hit, n_boot=BOOTSTRAP_SAMPLES, seed=SEED):
    hit = np.asarray(hit, float)
    draws = np.random.default_rng(seed).integers(0, len(hit), size=(n_boot, len(hit)))
    rates = hit[draws].mean(axis=1)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


# ---------------------------------------------------------------- fit worker (fresh process)

def _rss():
    return psutil.Process().memory_info().rss


def _maxrss_bytes():
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


class PeakRSSSampler:
    """Samples this process's resident set size on a background thread while a block runs."""

    def __init__(self, interval=RSS_SAMPLE_SECONDS):
        self.interval = interval
        self.peak = 0
        self.samples = 0
        self._stop = threading.Event()

    def _run(self):
        process = psutil.Process()
        while not self._stop.is_set():
            self.peak = max(self.peak, process.memory_info().rss)
            self.samples += 1
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = _rss()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, _rss())


def _timed(fn, repeats=TIMING_REPEATS):
    """Run fn several times; return the last result and the median wall-clock seconds."""
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        seconds.append(time.perf_counter() - start)
    return result, float(np.median(seconds)), seconds


def model_name(method, k, weighting):
    return f"{method}_{weighting}_k{k}"


def run_worker(method, k, weighting, models_dir=MODELS_DIR, processed_dir=PROCESSED_DIR):
    """One fit in this (fresh) process: memory and time of the fit, then the nightly steps."""
    sys.setswitchinterval(RSS_SAMPLE_SECONDS / 2)    # let the sampler thread run during the fit
    rss_after_imports = _rss()
    A = load_matrix(weighting, processed_dir)
    rss_loaded = _rss()
    maxrss_before = _maxrss_bytes()

    with PeakRSSSampler() as sampler:
        start = time.perf_counter()
        model = fit_decomposition(A, method, k)
        first_fit = time.perf_counter() - start
    maxrss_after = _maxrss_bytes()
    # Extra identical fits only for a stable median time (memory comes from the first fit above).
    fit_runs = [first_fit] + _timed(lambda: fit_decomposition(A, method, k), TIMING_REPEATS - 1)[2]
    fit_seconds = float(np.median(fit_runs))

    customer, product, mean = factor_arrays(model, A)
    P, normalize_seconds, _ = _timed(lambda: product_vectors(product))
    (item_idx, item_score), item_seconds, _ = _timed(lambda: item_neighbors(P))
    customer_recs, customer_seconds, _ = _timed(lambda: recommend_all_customers(A, P))

    if method == "svd":
        relative_error = relative_error_svd(A, model.components_)
        centred_error, variance = None, None
        spectrum = {"singular_values": model.singular_values_}
    else:
        relative_error, centred_error = relative_error_pca(A, model.components_, mean)
        variance = float(model.explained_variance_ratio_.sum())
        spectrum = {"explained_variance_ratio": model.explained_variance_ratio_, "mean": mean}

    models_dir.mkdir(parents=True, exist_ok=True)
    name = model_name(method, k, weighting)
    factors_path = models_dir / f"{name}_factors.npz"
    arrays = {"customer_factors": customer, "product_factors": product}
    if mean is not None:
        arrays["mean"] = mean
    np.savez(factors_path, **arrays)
    np.savez(models_dir / f"{name}_recs.npz", item_top_idx=item_idx, item_top_score=item_score,
             customer_top_idx=customer_recs, **spectrum)

    metrics = {
        "method": method, "k": k, "weighting": weighting, "seed": SEED,
        "solver": SOLVER if method == "svd" else "arpack",
        "matrix_sha256": matrix_fingerprint(processed_dir),
        "fit_fingerprint": fit_fingerprint(),
        "fit_seconds": fit_seconds,
        "fit_seconds_runs": fit_runs,
        "rss_before_imports_mb": _RSS_BEFORE_IMPORTS / MB,
        "rss_after_imports_mb": rss_after_imports / MB,
        "rss_loaded_matrix_mb": rss_loaded / MB,
        "peak_rss_mb": sampler.peak / MB,
        "rss_increase_mb": (sampler.peak - rss_loaded) / MB,
        "rss_samples": sampler.samples,
        "ru_maxrss_before_fit_mb": maxrss_before / MB,
        "ru_maxrss_after_fit_mb": maxrss_after / MB,
        "factor_mb": factor_bytes(customer, product, mean) / MB,
        "customer_factor_mb": customer.nbytes / MB,
        "product_factor_mb": product.nbytes / MB,
        "mean_vector_mb": (mean.nbytes / MB) if mean is not None else 0.0,
        "factor_file_mb": factors_path.stat().st_size / MB,
        "product_vectors_mb": P.nbytes / MB,
        "item_top10_table_mb": (item_idx.nbytes + item_score.nbytes) / MB,
        "relative_error": relative_error,
        "relative_error_centred": centred_error,
        "variance_explained": variance,
        "normalization_seconds": normalize_seconds,
        "item_scoring_seconds": item_seconds,
        "customer_scoring_seconds": customer_seconds,
    }
    (models_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def fit_fingerprint():
    """Hash of everything a cached fit depends on: the worker code (fitting, scoring, top-n padding,
    measurement), settings and library versions. The matrix contents are checked separately.
    Editing reports or figures does not invalidate the cached fits; editing any of these does."""
    parts = [inspect.getsource(fn) for fn in (
        load_quantity_matrix, load_matrix, weighted_matrix, purchase_value, fit_decomposition, factor_arrays,
        relative_error_svd, relative_error_pca, factor_bytes, product_vectors, top_n, item_neighbors, _mask_seen,
        customer_scores, recommend_all_customers, _rss, _maxrss_bytes, PeakRSSSampler, _timed, run_worker)]
    parts += [repr((SEED, SOLVER, TOP_N, TIMING_REPEATS, RSS_SAMPLE_SECONDS, MB)),
              np.__version__, scipy.__version__, sklearn.__version__, psutil.__version__, sys.version]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def worker_env():
    env = dict(os.environ)
    env.update({var: "1" for var in THREAD_VARS})
    return env


def ensure_fits(force=False, models_dir=MODELS_DIR, processed_dir=PROCESSED_DIR):
    """Run each fit in a fresh subprocess unless a cached result matches matrix and code."""
    matrix_hash = matrix_fingerprint(processed_dir)
    fingerprint = fit_fingerprint()
    metrics = {}
    for method, k, weighting in FIT_CONFIGS:
        name = model_name(method, k, weighting)
        path = models_dir / f"{name}.json"
        cached = json.loads(path.read_text()) if path.exists() else None
        if (force or cached is None or cached.get("matrix_sha256") != matrix_hash
                or cached.get("fit_fingerprint") != fingerprint):
            print(f"  fitting {name} in a fresh process")
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", method,
                            "--k", str(k), "--weighting", weighting], check=True,
                           env=worker_env(), cwd=ROOT)
            cached = json.loads(path.read_text())
        else:
            print(f"  reusing {name}")
        metrics[name] = cached
    return metrics


def load_fit(name, models_dir=MODELS_DIR):
    with np.load(models_dir / f"{name}_factors.npz") as f:
        factors = {key: f[key] for key in f.files}
    with np.load(models_dir / f"{name}_recs.npz") as f:
        recs = {key: f[key] for key in f.files}
    return factors, recs


# ---------------------------------------------------------------- component 1 and ML6 segments

def oriented(customer, product, component=0):
    """Flip one component so its product loadings sum to a positive number (signs are arbitrary)."""
    sign = 1.0 if product[:, component].sum() >= 0 else -1.0
    return customer[:, component] * sign, product[:, component] * sign


def describe_component(customer, product, Q, descriptions, component=0):
    """Measured meaning of one component: correlation with customer volume and top products."""
    scores, loadings = oriented(customer, product, component)
    log_volume = np.log1p(np.asarray(Q.sum(axis=1)).ravel())
    breadth = np.diff(Q.indptr)
    volume_corr = float(np.corrcoef(scores, log_volume)[0, 1])
    breadth_corr = float(np.corrcoef(scores, breadth)[0, 1])
    rank_corr = float(pd.Series(scores).corr(pd.Series(log_volume), method="spearman"))
    same_sign = float(100 * max((loadings > 0).mean(), (loadings < 0).mean()))
    top = np.argsort(-loadings, kind="stable")[:5]
    bottom = np.argsort(loadings, kind="stable")[:3]
    if volume_corr >= 0.6 and same_sign >= 90:
        words = (f"Overall buying volume: scores rise with how much and how widely a customer buys "
                 f"(r = {volume_corr:.2f} with log total quantity; {same_sign:.0f}% of product loadings share one sign)")
    else:
        words = (f"Contrast around the average customer: {descriptions[top[0]].title()} and similar "
                 f"best sellers versus {descriptions[bottom[0]].title()} "
                 f"(r = {volume_corr:.2f} with log total quantity; only {same_sign:.0f}% of loadings share one sign)")
    return {
        "description": words,
        "volume_correlation": volume_corr,
        "volume_spearman": rank_corr,
        "breadth_correlation": breadth_corr,
        "same_sign_loadings_pct": same_sign,
        "top_products": [descriptions[j] for j in top],
        "negative_products": [descriptions[j] for j in bottom],
    }


def map_coordinates(customer, product):
    """The first two customer components written in polar form.

    Component 1 is overall volume and is very skewed, so k-means on the raw pair isolates a
    handful of huge accounts. The same two numbers as (log distance from the origin, angle)
    separate how much a customer buys (size) from what they buy (mix): angle > 0 leans to
    component 2's positive products, angle < 0 to its negative ones.
    """
    c1 = oriented(customer, product, 0)[0]
    c2 = oriented(customer, product, 1)[0]
    size = np.log10(np.hypot(c1, c2))
    mix = np.degrees(np.arctan2(c2, c1))
    return np.column_stack([size, mix])


def cluster_customers(customer, product, n_clusters=N_SEGMENTS, seed=SEED):
    """k-means on the standardised map coordinates of the training-only SVD fit.

    Segment ids are renumbered from the largest to the smallest median size, so they do not
    depend on k-means' internal label order.
    """
    coords = map_coordinates(customer, product)
    z = StandardScaler().fit_transform(coords)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=KMEANS_N_INIT).fit(z)
    order = np.argsort([-np.median(coords[km.labels_ == c, 0]) for c in range(n_clusters)])
    relabel = np.empty(n_clusters, dtype=int)
    relabel[order] = np.arange(n_clusters)
    return coords, relabel[km.labels_]


# Keyword groups used only to put a plain product theme on each segment's distinctive products.
# More specific themes come first because ties go to the earlier theme.
PRODUCT_THEMES = {
    "Christmas": ("CHRISTMAS", "XMAS", "ADVENT", "SNOWFLAKE", "REINDEER", "SANTA", "BAUBLE"),
    "children's ranges": ("LUNCH BAG", "CHARLOTTE BAG", "SNACK BOX", "BACKPACK", "CHILDRENS", "KIDS", "TOY",
                          "DOLLY", "SPACEBOY", "CIRCUS", "SKIPPING", "PLAYHOUSE", "WOODLAND", "BEAKER",
                          "JIGSAW", "DINOSAUR", "GAME"),
    "kitchen & tea-ware": ("CAKE", "BAKING", "TEA", "MUG", "CUP", "PLATE", "BOWL", "KITCHEN", "JAM", "BOTTLE",
                         "SPOON", "RECIPE", "APRON", "NAPKIN"),
    "candles & lighting": ("CANDLE", "T-LIGHT", "TLIGHT", "LANTERN", "LIGHT"),
    "party & paper goods": ("PARTY", "BUNTING", "CARD", "PAPER", "WRAP", "STICKER", "BALLOON", "NOTEBOOK",
                            "PENCIL", "CHALK"),
    "home decor": ("SIGN", "HEART", "FRAME", "CLOCK", "CUSHION", "DOORMAT", "HOOK", "MIRROR", "WICKER",
                   "ORNAMENT", "HANGING", "CABINET", "DRAWER", "TRELLIS"),
    "bags & storage": ("BAG", "BOX", "STORAGE", "CASE", "BASKET", "JUMBO", "SHOPPER"),
    "garden & outdoor": ("GARDEN", "BIRD", "WATERING", "PICNIC", "PLANTER", "FLOWER"),
}


def product_theme(descriptions):
    """Most common theme among product descriptions (ties: order of PRODUCT_THEMES), used only when
    it matches at least half of them; otherwise "mixed gifts". A second theme is added when it
    matches at least two products and half as many as the first."""
    counts = {theme: sum(any(word in d.upper() for word in words) for d in descriptions)
              for theme, words in PRODUCT_THEMES.items()}
    ranked = sorted(counts, key=lambda t: -counts[t])
    first, second = ranked[0], ranked[1]
    if counts[first] == 0 or 2 * counts[first] < len(descriptions):
        return "mixed gifts"
    if counts[second] >= 2 and 2 * counts[second] >= counts[first]:
        return f"{first}, {second}"
    return first


def mix_lift(binary, members, min_share=0.02, min_buyers=10):
    """How over-represented each product is in a segment's baskets, independent of volume:
    (product's share of the segment's purchases) / (its share of all purchases).
    Products bought by fewer than max(min_buyers, min_share x segment size) members get 0."""
    seg_buyers = np.asarray(binary[members].sum(axis=0)).ravel()
    all_buyers = np.asarray(binary.sum(axis=0)).ravel()
    lift = (seg_buyers / seg_buyers.sum()) / np.maximum(all_buyers / all_buyers.sum(), 1e-12)
    floor = max(min_buyers, min_share * members.sum())
    return np.where(seg_buyers >= floor, lift, 0.0)


def segment_profiles(labels, Q, countries, descriptions, stock_codes, invoices_per_customer):
    """Describe each segment with training-only volume, breadth, top and distinctive products,
    and the share of its customers whose dominant (training) country is each country."""
    binary = (Q > 0).astype(np.float64).tocsr()
    volume = np.asarray(Q.sum(axis=1)).ravel()
    breadth = np.diff(Q.indptr)
    overall_median_volume = np.median(volume)
    overall_uk = float(np.mean(countries == "United Kingdom"))
    profiles = []
    for seg in range(labels.max() + 1):
        members = labels == seg
        penetration = np.asarray(binary[members].mean(axis=0)).ravel()
        top = np.argsort(-penetration, kind="stable")[:5]
        lift = mix_lift(binary, members)
        distinctive = np.argsort(-lift, kind="stable")[:10]
        country_share = pd.Series(countries[members]).value_counts(normalize=True)
        profiles.append({
            "segment_id": seg,
            "customers": int(members.sum()),
            "customer_share_pct": 100 * members.mean(),
            "total_quantity": float(volume[members].sum()),
            "avg_quantity_per_customer": float(volume[members].mean()),
            "median_quantity_per_customer": float(np.median(volume[members])),
            "median_volume_vs_all": float(np.median(volume[members]) / overall_median_volume),
            "avg_distinct_products": float(breadth[members].mean()),
            "avg_invoices": float(invoices_per_customer[members].mean()),
            "top_products": [
                {"stock_code": stock_codes[j], "description": descriptions[j],
                 "buyer_share_pct": 100 * penetration[j]} for j in top],
            "distinctive_products": [
                {"stock_code": stock_codes[j], "description": descriptions[j],
                 "buyer_share_pct": 100 * penetration[j], "mix_lift": float(lift[j])} for j in distinctive],
            "countries": [{"country": c, "customer_share_pct": 100 * s}
                          for c, s in country_share.head(3).items()],
            "uk_share_pct": 100 * float(country_share.get("United Kingdom", 0.0)),
            "overall_uk_share_pct": 100 * overall_uk,
        })
    _name_segments(profiles)
    return profiles


def volume_tier(median_volume_vs_all):
    """Plain-words size band from the segment's median training quantity / all customers' median."""
    if median_volume_vs_all < 0.33:
        return "Occasional small buyers"
    if median_volume_vs_all < 0.8:
        return "Smaller regular buyers"
    if median_volume_vs_all < 2:
        return "Mid-size regular buyers"
    return "High-volume wholesalers"


def _name_segments(profiles):
    """Name = volume tier + theme of the segment's ten most distinctive products (or "mixed gifts"
    when no theme covers half of them), plus an
    'international' tag only when the non-UK share is at least double the overall non-UK share."""
    for p in profiles:
        theme = product_theme([d["description"] for d in p["distinctive_products"]])
        non_uk, overall_non_uk = 100 - p["uk_share_pct"], 100 - p["overall_uk_share_pct"]
        geo = ", more international" if non_uk >= 2 * overall_non_uk else ""
        p["theme"] = theme
        p["name"] = f"{volume_tier(p['median_volume_vs_all'])}: {theme}{geo}"


def segments_table(profiles):
    return pd.DataFrame([{
        "segment_id": p["segment_id"],
        "name": p["name"],
        "customers": p["customers"],
        "customer_share_pct": round(p["customer_share_pct"], 1),
        "top_products": "; ".join(f"{d['description']} ({d['buyer_share_pct']:.0f}%)" for d in p["top_products"]),
        "distinctive_products": "; ".join(f"{d['description']} (mix lift {d['mix_lift']:.1f})" for d in p["distinctive_products"][:5]),
        "countries": "; ".join(f"{c['country']} {c['customer_share_pct']:.1f}%" for c in p["countries"]),
        "total_quantity": int(p["total_quantity"]),
        "avg_quantity_per_customer": round(p["avg_quantity_per_customer"], 1),
        "median_quantity_per_customer": round(p["median_quantity_per_customer"], 1),
        "avg_distinct_products": round(p["avg_distinct_products"], 1),
        "avg_invoices": round(p["avg_invoices"], 1),
    } for p in profiles])


# ---------------------------------------------------------------- figures

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_svd_curve(curve, chosen_k, sparse_mb, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    ax, ax2 = axes
    ax.plot(curve["k"], curve["relative_error"], color=SERIES[0], marker="o", linewidth=2, markersize=8)
    for k, e in zip(curve["k"], curve["relative_error"]):
        ax.annotate(f"{e:.3f}", (k, e), textcoords="offset points", xytext=(6, 6), color=INK, fontsize=9)
    ax.set_ylim(curve["relative_error"].min() - 0.03, curve["relative_error"].max() + 0.04)
    ax.set_title("Relative reconstruction error, training matrix", color=INK, loc="left", fontsize=11)
    ax.set_ylabel("||A - A_k||_F / ||A||_F", color=INK_2)
    ax2.plot(curve["k"], curve["factor_mb"], color=SERIES[0], marker="o", linewidth=2, markersize=8)
    for k, mb in zip(curve["k"], curve["factor_mb"]):
        ax2.annotate(f"{mb:.1f}", (k, mb), textcoords="offset points", xytext=(0, 10), ha="center", color=INK, fontsize=9)
    ax2.axhline(FACTOR_TARGET_MB, color=INK_2, linestyle=":", linewidth=1.2)
    ax2.text(curve["k"].min(), FACTOR_TARGET_MB, f"{FACTOR_TARGET_MB:.0f} MB target", va="bottom", color=INK_2, fontsize=9)
    ax2.axhline(sparse_mb, color=INK_2, linestyle="--", linewidth=1)
    ax2.text(curve["k"].min(), sparse_mb, f"sparse matrix {sparse_mb:.1f} MB", va="bottom", color=INK_2, fontsize=9)
    ax2.set_ylim(0, curve["factor_mb"].max() * 1.15)
    ax2.set_title("Stored factors, customers + products (MB)", color=INK, loc="left", fontsize=11)
    for a in axes:
        _style(a)
        a.axvline(chosen_k, color=INK_2, linestyle="--", linewidth=1)
        a.text(chosen_k, a.get_ylim()[1], f" chosen k = {chosen_k}", color=INK_2, fontsize=9, va="top")
        a.set_xscale("log")
        a.set_xticks(curve["k"], [str(k) for k in curve["k"]])
        a.minorticks_off()
        a.set_xlabel("k, number of components (log scale)", color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_customer_map(coords, labels, profiles, comp1, comp2, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7.4), facecolor=SURFACE)
    _style(ax)
    for p in profiles:
        seg = p["segment_id"]
        pts = coords[labels == seg]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, marker=MARKERS[seg], color=SERIES[seg], alpha=0.55,
                   linewidths=0, label=f"{p['name']} ({p['customers']:,} customers, {p['customer_share_pct']:.0f}%)")
    for p in profiles:
        cx, cy = np.median(coords[labels == p["segment_id"]], axis=0)
        ax.annotate(p["name"].split(":")[0], (cx, cy), color=INK, fontsize=9, ha="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=INK_2, lw=0.6, alpha=0.9))
    _full_extent(ax, coords[:, 0], coords[:, 1])
    up, down = comp2["top_products"][0].title(), comp2["negative_products"][0].title()
    ax.set_xlabel(f"Size: log10 distance from the origin on components 1-2 (component 1 = buying volume, "
                  f"r = {comp1['volume_correlation']:.2f} with log quantity)", color=INK_2, fontsize=9)
    ax.set_ylabel(f"Mix: angle in degrees (up = more like {up};\ndown = more like {down})", color=INK_2, fontsize=9)
    ax.set_title("Customer map: four segments from the first two SVD components (training data only)",
                 color=INK, loc="left", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.1), ncol=1, frameon=False, fontsize=9,
              labelcolor=INK, markerscale=1.8)
    fig.text(0.01, 0.01, f"All {len(coords):,} customers shown, one anonymous point each (no IDs). k-means was fitted "
             "on these two training-only values, standardised;\nit does not use the other 98 components.",
             color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _full_extent(ax, x, y, pad=0.04, log_x=False):
    """Axis limits covering every point, plus a small margin (no percentile cropping)."""
    if log_x:
        lo, hi = np.log10(x.min()), np.log10(x.max())
        ax.set_xlim(10 ** (lo - pad * (hi - lo)), 10 ** (hi + pad * (hi - lo)))
    else:
        ax.set_xlim(x.min() - pad * np.ptp(x), x.max() + pad * np.ptp(x))
    ax.set_ylim(y.min() - pad * np.ptp(y), y.max() + pad * np.ptp(y))


def plot_customer_components(customer, product, labels, profiles, path):
    """Supplementary view: the literal (oriented) component 1 and 2 scores, coloured by the same
    segment labels as the polar map. Left: linear axes. Right: the same points, component 1 on a log axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c1 = oriented(customer, product, 0)[0]
    c2 = oriented(customer, product, 1)[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), facecolor=SURFACE)
    for ax, log_x in zip(axes, (False, True)):
        _style(ax)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        for p in profiles:
            seg = p["segment_id"]
            members = labels == seg
            ax.scatter(c1[members], c2[members], s=12, marker=MARKERS[seg], color=SERIES[seg], alpha=0.55,
                       linewidths=0, label=f"{p['name']} ({p['customers']:,} customers)")
        if log_x:
            ax.set_xscale("log")
        _full_extent(ax, c1, c2, log_x=log_x)
        ax.set_ylabel("Component 2 score", color=INK_2, fontsize=9)
    axes[0].set_xlabel("Component 1 score (linear axis)", color=INK_2, fontsize=9)
    axes[1].set_xlabel("Component 1 score (log axis, same values)", color=INK_2, fontsize=9)
    axes[0].set_title("Literal scores, full range", color=INK, loc="left", fontsize=10)
    axes[1].set_title("Same points, component 1 on a log axis", color=INK, loc="left", fontsize=10)
    fig.suptitle("Supplementary: first two training SVD component scores, coloured by the customer-map segments",
                 color=INK, x=0.01, ha="left", fontsize=11)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower left", bbox_to_anchor=(0.01, 0.06), ncol=2, frameon=False, fontsize=9,
               labelcolor=INK, markerscale=1.8)
    fig.text(0.01, 0.01, f"All {len(c1):,} customers shown, one anonymous point each (no IDs). Segments were fitted by "
             "k-means on the standardised (log10 radius, angle) of these two scores,\nnot on all 100 components. "
             "Component signs are oriented so product loadings sum to a positive number.", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_evaluation(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    rows = rows[::-1]
    rates = np.array([100 * r["hit_rate_at_10"] for r in rows])
    low = np.array([100 * r["ci95_low"] for r in rows])
    high = np.array([100 * r["ci95_high"] for r in rows])
    fill = {"main": SERIES[0], "model": "#86b6ef", "baseline": GRID}
    edge = {"main": SERIES[0], "model": "#86b6ef", "baseline": INK_2}
    y = np.arange(len(rows))
    ax.barh(y, rates, color=[fill[r["kind"]] for r in rows], edgecolor=[edge[r["kind"]] for r in rows], height=0.6)
    ax.errorbar(rates, y, xerr=[rates - low, high - rates], fmt="none", ecolor=INK, capsize=3, linewidth=1)
    for yi, rate, hi_ in zip(y, rates, high):
        ax.text(hi_ + 0.6, yi, f"{rate:.1f}%", va="center", color=INK, fontsize=9)
    base = [r for r in rows if r["kind"] == "baseline"][0]
    ax.axvline(100 * base["hit_rate_at_10"], color=INK_2, linestyle=":", linewidth=1)
    ax.set_yticks(y, [r["label"] for r in rows], color=INK)
    ax.set_xlim(0, max(high) * 1.18)
    ax.set_xlabel("Hit rate at 10 on held-out new purchases (%), with 95% customer bootstrap interval", color=INK_2)
    n = rows[0]["evaluated_customers"]
    ax.set_title(f"Same {n:,} warm test customers and catalogue in every row\n"
                 "count and binary rows are the ML7 sensitivity check (not re-selected)",
                 color=INK, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- environment and JSON

def environment_info():
    sysctl = "/usr/sbin/sysctl" if os.path.exists("/usr/sbin/sysctl") else "sysctl"    # not always on PATH
    try:
        cpu = subprocess.run([sysctl, "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor() or platform.machine()
    blas = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {}).get("name")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "logical_cpus": psutil.cpu_count(),
        "ram_gb": psutil.virtual_memory().total / 1e9,
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "psutil": psutil.__version__,
        "numpy_blas": blas,
        "thread_env": {var: os.environ.get(var) for var in THREAD_VARS},
        "memory_sampling": (f"psutil RSS sampled every {RSS_SAMPLE_SECONDS * 1000:.0f} ms on a background "
                            "thread during the first fit in a fresh process; ru_maxrss high-water mark kept as a check"),
        "timing": f"median of {TIMING_REPEATS} runs (time.perf_counter), one thread",
    }


def _jsonable(value):
    # Strict JSON has no NaN/Infinity, so non-finite floats (e.g. the last k-curve step,
    # which has no next k) are written as null.
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def write_json(obj, path):
    Path(path).write_text(json.dumps(_jsonable(obj), indent=2, allow_nan=False) + "\n")


# ---------------------------------------------------------------- main analysis

def run_analysis(force=False):
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    matrix_report = json.loads((REPORTS_DIR / "matrix_report.json").read_text())

    print("ML2/ML5/ML7 fits (one fresh process per fit)")
    fits = ensure_fits(force)
    chosen = model_name("svd", CHOSEN_K, MAIN_WEIGHTING)
    pca_name = model_name("pca", CHOSEN_K, MAIN_WEIGHTING)

    print("Loading training data and held-out purchases")
    Q = load_quantity_matrix()
    A = load_matrix(MAIN_WEIGHTING)
    products = pd.read_csv(PROCESSED_DIR / "product_index.csv", dtype={"stock_code": str})
    customer_ids = pd.read_csv(PROCESSED_DIR / "customer_index.csv")["customer_id"].to_numpy()
    countries = pd.read_csv(PROCESSED_DIR / "customer_countries.csv")["country"].to_numpy()
    stock_codes, descriptions = products["stock_code"].to_numpy(), products["description"].to_numpy()
    train = pd.read_csv(PROCESSED_DIR / "train.csv", usecols=["Invoice", "Customer ID"], dtype={"Invoice": str})
    invoices = train.groupby("Customer ID")["Invoice"].nunique().reindex(customer_ids).to_numpy()
    test = pd.read_csv(PROCESSED_DIR / "test.csv", dtype={"StockCode": str, "Invoice": str})

    # ML2 curve: training reconstruction error and storage only.
    curve = pd.DataFrame([{
        "k": k,
        "relative_error": fits[model_name("svd", k, MAIN_WEIGHTING)]["relative_error"],
        "fit_seconds": fits[model_name("svd", k, MAIN_WEIGHTING)]["fit_seconds"],
        "peak_rss_mb": fits[model_name("svd", k, MAIN_WEIGHTING)]["peak_rss_mb"],
        "factor_mb": fits[model_name("svd", k, MAIN_WEIGHTING)]["factor_mb"],
    } for k in K_SWEEP])
    # Error removed per added component between consecutive k values (positive = error falls).
    curve["error_drop_per_component"] = (curve["relative_error"].diff(-1) / -curve["k"].diff(-1)).round(6)
    curve["svd_share_of_squared_norm_kept"] = 1 - curve["relative_error"] ** 2
    _, pca_big = load_fit(model_name("pca", max(K_SWEEP), MAIN_WEIGHTING))
    curve["pca_variance_explained"] = [pca_big["explained_variance_ratio"][:k].sum() for k in K_SWEEP]
    curve["within_factor_target"] = curve["factor_mb"] <= FACTOR_TARGET_MB
    curve.to_csv(REPORTS_DIR / "svd_curve.csv", index=False)
    plot_svd_curve(curve, CHOSEN_K, matrix_report["sparse_mb"], FIGURES_DIR / "svd_curve.png")

    print("ML4 evaluation on held-out purchases")
    targets, coverage = heldout_targets(test, customer_ids, stock_codes)
    totals = best_seller_totals(Q)
    baseline_recs = best_seller_recommendations(A, totals)
    baseline = evaluate(baseline_recs, targets, coverage)
    results_by_model = {}
    for name in fits:
        _, recs = load_fit(name)
        results_by_model[name] = evaluate(recs["customer_top_idx"], targets, coverage)

    # Deployed view: cold customers get the overall best sellers.
    top_overall = top_n(totals.astype(float))[0][0]
    top_overall = top_overall[top_overall >= 0]          # drop padding (only if the catalogue had < 10)
    test_pairs = test[["Customer ID", "StockCode"]].drop_duplicates()
    cold = test_pairs.loc[~test_pairs["Customer ID"].isin(customer_ids)]
    cold_hits = int(cold.groupby("Customer ID")["StockCode"]
                    .apply(lambda s: s.isin(set(stock_codes[top_overall])).any()).sum())

    svd_eval, pca_eval = results_by_model[chosen], results_by_model[pca_name]
    comparisons = {
        "svd_vs_baseline": paired_comparison(svd_eval["per_customer_hit"], baseline["per_customer_hit"]),
        "pca_vs_baseline": paired_comparison(pca_eval["per_customer_hit"], baseline["per_customer_hit"]),
        "svd_vs_pca": paired_comparison(svd_eval["per_customer_hit"], pca_eval["per_customer_hit"]),
    }

    # Component 1 in words, for both methods (training factors only).
    svd_factors, svd_recs = load_fit(chosen)
    pca_factors, _ = load_fit(pca_name)
    comp_svd = describe_component(svd_factors["customer_factors"], svd_factors["product_factors"], Q, descriptions)
    comp_pca = describe_component(pca_factors["customer_factors"], pca_factors["product_factors"], Q, descriptions)

    print("ML6 customer segments")
    coords, labels = cluster_customers(svd_factors["customer_factors"], svd_factors["product_factors"])
    profiles = segment_profiles(labels, Q, countries, descriptions, stock_codes, invoices)
    comp2 = describe_component(svd_factors["customer_factors"], svd_factors["product_factors"], Q, descriptions, 1)
    seg_table = segments_table(profiles)
    seg_table.to_csv(REPORTS_DIR / "segments.csv", index=False)
    write_json({"method": f"k-means (k={N_SEGMENTS}, n_init={KMEANS_N_INIT}, random_state={SEED}) on the "
                          "standardised (log10 radius, angle) of the first two customer components of the chosen "
                          "training-only SVD fit (not all 100 components)",
                "component_2": comp2, "segments": profiles}, REPORTS_DIR / "segment_profiles.json")
    plot_customer_map(coords, labels, profiles, comp_svd, comp2, FIGURES_DIR / "customer_map.png")
    plot_customer_components(svd_factors["customer_factors"], svd_factors["product_factors"], labels, profiles,
                             FIGURES_DIR / "customer_map_components.png")

    print("ML7 purchase-value sensitivity")
    sensitivity = []
    for weighting in WEIGHTINGS:
        name = model_name("svd", CHOSEN_K, weighting)
        ev, fm = results_by_model[name], fits[name]
        sensitivity.append({
            "weighting": weighting, "k": CHOSEN_K, "main_model": weighting == MAIN_WEIGHTING,
            "hit_rate_at_10": ev["hit_rate_at_10"], "hits": ev["hits"],
            "evaluated_customers": ev["evaluated_customers"], "recall_at_10": ev["recall_at_10"],
            "relative_error": fm["relative_error"], "fit_seconds": fm["fit_seconds"],
            **{f"vs_baseline_{key}": val for key, val in
               paired_comparison(ev["per_customer_hit"], baseline["per_customer_hit"]).items()
               if key in ("difference", "ci95_low", "ci95_high")},
        })
    pd.DataFrame(sensitivity).to_csv(REPORTS_DIR / "sensitivity.csv", index=False)

    # Descriptive only: test hit rate across the k sweep. k was fixed before this was computed.
    k_descriptive = pd.DataFrame([{
        "k": k, "hit_rate_at_10": results_by_model[model_name("svd", k, MAIN_WEIGHTING)]["hit_rate_at_10"],
        "hits": results_by_model[model_name("svd", k, MAIN_WEIGHTING)]["hits"],
        "evaluated_customers": coverage["eligible_customers"],
        "note": "descriptive only; not used to choose k",
    } for k in K_SWEEP])
    k_descriptive.to_csv(REPORTS_DIR / "k_test_descriptive.csv", index=False)

    svd_fit, pca_fit = fits[chosen], fits[pca_name]
    nightly = {
        "fit_seconds": svd_fit["fit_seconds"],
        "normalization_seconds": svd_fit["normalization_seconds"],
        "item_scoring_seconds": svd_fit["item_scoring_seconds"],
        "total_seconds": svd_fit["fit_seconds"] + svd_fit["normalization_seconds"] + svd_fit["item_scoring_seconds"],
        "customer_scoring_seconds": svd_fit["customer_scoring_seconds"],
        "target_seconds": NIGHTLY_TARGET_SECONDS,
        "products_scored": int(A.shape[1]),
        "customers_scored": int(A.shape[0]),
        "includes": "fit + normalise product vectors + top-10 neighbours for every catalogue product, on the "
                    "prepared in-memory matrix; personalised top-10 for every customer is timed separately",
        "excludes": "raw Excel reading and cleaning, Python imports, loading and saving files, deployment and network",
    }

    def model_block(fit, ev, comp, cmp):
        ci = bootstrap_interval(ev["per_customer_hit"])
        return {
            "k": fit["k"], "fit_seconds": fit["fit_seconds"], "fit_seconds_runs": fit["fit_seconds_runs"],
            "peak_rss_mb": fit["peak_rss_mb"], "rss_increase_mb": fit["rss_increase_mb"],
            "rss_after_imports_mb": fit["rss_after_imports_mb"], "rss_loaded_matrix_mb": fit["rss_loaded_matrix_mb"],
            "ru_maxrss_after_fit_mb": fit["ru_maxrss_after_fit_mb"],
            "factor_mb": fit["factor_mb"], "customer_factor_mb": fit["customer_factor_mb"],
            "product_factor_mb": fit["product_factor_mb"], "mean_vector_mb": fit["mean_vector_mb"],
            "factor_file_mb": fit["factor_file_mb"], "product_vectors_mb": fit["product_vectors_mb"],
            "item_top10_table_mb": fit["item_top10_table_mb"],
            "relative_error": fit["relative_error"], "relative_error_centred": fit["relative_error_centred"],
            "variance_explained": fit["variance_explained"],
            "hit_rate_at_10": ev["hit_rate_at_10"], "hits": ev["hits"], "evaluated_customers": ev["evaluated_customers"],
            "hit_rate_ci95_low": ci[0], "hit_rate_ci95_high": ci[1],
            "recall_at_10": ev["recall_at_10"], "warm_hit_rate_at_10": ev["warm_hit_rate_at_10"],
            "all_test_hit_rate_at_10": ev["all_test_hit_rate_at_10"],
            "vs_baseline": cmp,
            "component_1_description": comp["description"],
            "component_1_volume_correlation": comp["volume_correlation"],
            "component_1_volume_spearman": comp["volume_spearman"],
            "component_1_breadth_correlation": comp["breadth_correlation"],
            "component_1_same_sign_loadings_pct": comp["same_sign_loadings_pct"],
            "component_1_top_products": comp["top_products"],
            "solver": fit["solver"],
        }

    svd_block = model_block(svd_fit, svd_eval, comp_svd, comparisons["svd_vs_baseline"])
    pca_block = model_block(pca_fit, pca_eval, comp_pca, comparisons["pca_vs_baseline"])
    pca_block["centring"] = "implicit (sparse input, svd_solver='arpack'); no scaling"
    base_ci = bootstrap_interval(baseline["per_customer_hit"])
    baseline_block = {
        "hit_rate_at_10": baseline["hit_rate_at_10"], "hits": baseline["hits"],
        "evaluated_customers": baseline["evaluated_customers"], "recall_at_10": baseline["recall_at_10"],
        "hit_rate_ci95_low": base_ci[0], "hit_rate_ci95_high": base_ci[1],
        "warm_hit_rate_at_10": baseline["warm_hit_rate_at_10"],
        "all_test_hit_rate_at_10": baseline["all_test_hit_rate_at_10"],
        "ranking": "training quantity summed per catalogue product; each customer's already-bought products removed and refilled",
        "top_10_products": [descriptions[j] for j in top_overall],
    }

    # ML5 comparison table (the eight required measures plus supporting rows).
    comparison = pd.DataFrame([
        ("k used", pca_block["k"], svd_block["k"]),
        ("Fit time (seconds)", round(pca_block["fit_seconds"], 3), round(svd_block["fit_seconds"], 3)),
        ("Peak memory during fit (MB, process RSS)", round(pca_block["peak_rss_mb"], 1), round(svd_block["peak_rss_mb"], 1)),
        ("Memory increase during fit (MB, above loaded matrix)", round(pca_block["rss_increase_mb"], 1), round(svd_block["rss_increase_mb"], 1)),
        ("Size of stored factors (MB, customers + products [+ mean])", round(pca_block["factor_mb"], 3), round(svd_block["factor_mb"], 3)),
        ("Variance explained (PCA) / relative reconstruction error (SVD)",
         f"{100 * pca_block['variance_explained']:.1f}% variance explained",
         f"{svd_block['relative_error']:.4f} relative error"),
        ("Relative reconstruction error of A (same formula for both)", round(pca_block["relative_error"], 4), round(svd_block["relative_error"], 4)),
        ("Hit rate at 10 on held-out purchases", f"{100 * pca_block['hit_rate_at_10']:.1f}% ({pca_block['hits']}/{pca_block['evaluated_customers']})",
         f"{100 * svd_block['hit_rate_at_10']:.1f}% ({svd_block['hits']}/{svd_block['evaluated_customers']})"),
        ("Best-seller baseline hit rate at 10", f"{100 * baseline_block['hit_rate_at_10']:.1f}% ({baseline_block['hits']}/{baseline_block['evaluated_customers']})",
         f"{100 * baseline_block['hit_rate_at_10']:.1f}% ({baseline_block['hits']}/{baseline_block['evaluated_customers']})"),
        ("What component 1 represents, in words", comp_pca["description"], comp_svd["description"]),
    ], columns=["measure", "PCA", "Truncated SVD"])
    comparison.to_csv(REPORTS_DIR / "comparison.csv", index=False)

    def eval_row(label, kind, ev):
        low, high = bootstrap_interval(ev["per_customer_hit"])
        return {"label": label, "kind": kind, "hit_rate_at_10": ev["hit_rate_at_10"], "hits": ev["hits"],
                "evaluated_customers": ev["evaluated_customers"], "recall_at_10": ev["recall_at_10"],
                "ci95_low": low, "ci95_high": high}

    eval_rows = [
        eval_row(f"SVD k={CHOSEN_K}, log (main)", "main", svd_eval),
        eval_row(f"PCA k={CHOSEN_K}, log", "model", pca_eval),
        *[eval_row(f"SVD k={CHOSEN_K}, {w}", "model", results_by_model[model_name("svd", CHOSEN_K, w)])
          for w in WEIGHTINGS if w != MAIN_WEIGHTING],
        eval_row("Best sellers (baseline)", "baseline", baseline),
    ]
    plot_evaluation(eval_rows, FIGURES_DIR / "evaluation.png")
    pd.DataFrame(eval_rows).rename(columns={"label": "model"}).to_csv(REPORTS_DIR / "evaluation.csv", index=False)

    evaluation = {
        **coverage,
        "primary_cohort": "warm matrix customers with at least one held-out product in the training catalogue",
        "coverage_definitions": {
            "eligible_customer_coverage_pct": "eligible customers / all distinct test customers",
            "eligible_pair_coverage_pct": "held-out (customer, product) pairs with a warm customer and a catalogue "
                                          "product / all distinct test pairs",
            "eligible_product_coverage_pct": "distinct test products in the training catalogue / all distinct test products",
        },
        "recall_note": "recall_at_10 = mean over cohort customers of (held-out products in the top 10) / "
                       "(that customer's held-out catalogue products); the denominator is not capped at 10",
        "warm_rate_note": "warm_hit_rate_at_10 divides the same hits by every warm test customer, "
                          "so customers whose held-out products are all outside the catalogue count as misses",
        "all_test_rate_note": "all_test_hit_rate_at_10 divides by every test customer, cold-start customers counted as misses",
        "cold_start_bestseller_hits": cold_hits,
        "svd_all_test_hit_rate_with_cold_fallback": (svd_eval["hits"] + cold_hits) / coverage["total_test_customers"],
        "baseline_all_test_hit_rate_with_cold_fallback": (baseline["hits"] + cold_hits) / coverage["total_test_customers"],
        "comparisons": comparisons,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }

    results = {
        "schema_version": "1.0",
        "generated_by": "model_analysis.py",
        "chosen_model": "Truncated SVD",
        "chosen_k": CHOSEN_K,
        "seed": SEED,
        "purchase_value": MAIN_WEIGHTING,
        "top_n": TOP_N,
        "memory_units": "decimal MB (1 MB = 1,000,000 bytes)",
        "rate_units": "*_rate* and hit_rate_at_10 are fractions in 0..1; *_pct fields are percentages in 0..100",
        "matrix": {
            "customers": matrix_report["customers"], "products": matrix_report["products"],
            "nonzeros": matrix_report["nonzeros"], "sparsity_pct": matrix_report["sparsity_pct"],
            "dense_mb": matrix_report["dense_mb"], "sparse_mb": matrix_report["sparse_mb"],
            "min_buyers_per_product": matrix_report["min_buyers_per_product"],
        },
        "data": {key: matrix_report[key] for key in (
            "raw_rows", "clean_rows", "train_rows", "test_rows", "excluded_repeat_rows", "cutoff",
            "train_start", "train_end", "test_start", "test_end")},
        "evaluation": evaluation,
        "svd": svd_block,
        "pca": pca_block,
        "baseline": baseline_block,
        "nightly": nightly,
        "sensitivity": sensitivity,
        "k_curve": curve.to_dict(orient="records"),
        "k_choice": {
            "chosen_k": CHOSEN_K,
            "reason": "Predeclared in the partner's ML2 and kept: the training error still falls at k=100 "
                      "but each added component buys less, and k=100 is the largest swept k whose factors "
                      f"stay under the {FACTOR_TARGET_MB:.0f} MB storage target (k=200 does not). "
                      "Test results were not used.",
        },
        "segments": [{
            "segment_id": p["segment_id"], "name": p["name"], "customers": p["customers"],
            "customer_share_pct": p["customer_share_pct"], "theme": p["theme"],
            "top_products": [d["description"] for d in p["top_products"]],
            "distinctive_products": [d["description"] for d in p["distinctive_products"]],
            "countries": p["countries"], "avg_quantity_per_customer": p["avg_quantity_per_customer"],
            "median_quantity_per_customer": p["median_quantity_per_customer"],
            "avg_distinct_products": p["avg_distinct_products"], "avg_invoices": p["avg_invoices"],
        } for p in profiles],
        "component_2": {key: comp2[key] for key in ("volume_correlation", "top_products", "negative_products")},
        "targets": {
            "factor_mb_max": FACTOR_TARGET_MB, "factor_target_met": svd_block["factor_mb"] <= FACTOR_TARGET_MB,
            "nightly_seconds_max": NIGHTLY_TARGET_SECONDS, "nightly_target_met": nightly["total_seconds"] <= NIGHTLY_TARGET_SECONDS,
            "beats_baseline": svd_block["hit_rate_at_10"] > baseline_block["hit_rate_at_10"],
            "segments": len(profiles), "segments_target_met": len(profiles) == N_SEGMENTS,
        },
        "files": {
            "reports": ["data_profile.csv", "sheet_counts.csv", "cleaning_log.csv", "split_log.csv",
                        "matrix_report.json", "svd_curve.csv", "comparison.csv", "evaluation.csv",
                        "sensitivity.csv", "segments.csv", "segment_profiles.json", "k_test_descriptive.csv",
                        "results.json"],
            "figures": ["svd_curve.png", "customer_map.png", "customer_map_components.png", "evaluation.png"],
        },
        "environment": environment_info(),
    }
    write_json(results, REPORTS_DIR / "results.json")

    print(f"  SVD k={CHOSEN_K}: hit rate {svd_block['hit_rate_at_10']:.3f} "
          f"({svd_block['hits']}/{svd_block['evaluated_customers']}), PCA {pca_block['hit_rate_at_10']:.3f}, "
          f"best sellers {baseline_block['hit_rate_at_10']:.3f}")
    print(f"  nightly job {nightly['total_seconds']:.2f} s, factors {svd_block['factor_mb']:.3f} MB")
    print(f"  wrote reports/results.json and {len(results['files']['figures'])} figures")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="refit every model even if cached results match")
    parser.add_argument("--worker", choices=["svd", "pca"], help=argparse.SUPPRESS)
    parser.add_argument("--k", type=int, default=CHOSEN_K, help=argparse.SUPPRESS)
    parser.add_argument("--weighting", choices=WEIGHTINGS, default=MAIN_WEIGHTING, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        m = run_worker(args.worker, args.k, args.weighting)
        print(f"    fit {m['fit_seconds']:.3f} s, peak RSS {m['peak_rss_mb']:.1f} MB "
              f"(+{m['rss_increase_mb']:.1f} MB), relative error {m['relative_error']:.4f}")
        return
    run_analysis(force=args.force)


if __name__ == "__main__":
    main()
