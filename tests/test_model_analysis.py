"""ML checks against independent answers: dense numpy reconstructions, brute-force loops and
hand-counted hit rates on tiny fixtures."""

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD

import model_analysis as ma


@pytest.fixture
def small_matrix():
    rng = np.random.default_rng(0)
    dense = rng.poisson(0.4, size=(30, 12)).astype(float)
    dense[dense > 0] = np.log1p(dense[dense > 0])
    dense[0, 0] = 2.0
    return sp.csr_matrix(dense)


def test_sparse_svd_error_matches_dense_reconstruction(small_matrix):
    dense = small_matrix.toarray()
    for k in (2, 5):
        svd = TruncatedSVD(k, algorithm="arpack", random_state=42).fit(small_matrix)
        U, S, Vt = np.linalg.svd(dense, full_matrices=False)
        exact = np.linalg.norm(dense - (U[:, :k] * S[:k]) @ Vt[:k]) / np.linalg.norm(dense)
        assert ma.relative_error_svd(small_matrix, svd.components_) == pytest.approx(exact, rel=1e-6)


def test_sparse_pca_error_matches_dense_reconstruction(small_matrix):
    dense = small_matrix.toarray()
    pca = PCA(4, svd_solver="arpack", random_state=42).fit(small_matrix)
    mean, W = dense.mean(axis=0), pca.components_
    rebuilt = mean + (dense - mean) @ W.T @ W
    residual = np.linalg.norm(dense - rebuilt)
    vs_a, vs_centred = ma.relative_error_pca(small_matrix, W, pca.mean_)
    assert vs_a == pytest.approx(residual / np.linalg.norm(dense), rel=1e-6)
    assert vs_centred == pytest.approx(residual / np.linalg.norm(dense - mean), rel=1e-6)
    # Centred error and variance explained describe the same thing.
    assert vs_centred ** 2 == pytest.approx(1 - pca.explained_variance_ratio_.sum(), rel=1e-6)


def test_top_n_breaks_ties_by_index_and_pads_excluded():
    scores = np.array([[0.5, 0.9, 0.5, 0.9, -np.inf, -np.inf]])
    idx, best = ma.top_n(scores, 5)
    assert idx[0].tolist() == [1, 3, 0, 2, -1]
    assert best[0, :4].tolist() == [0.9, 0.9, 0.5, 0.5]


def test_recommend_similar_is_brute_force_cosine_without_self():
    rng = np.random.default_rng(1)
    factors = rng.normal(size=(15, 3))
    P = ma.product_vectors(factors)
    idx, _ = ma.recommend_similar(4, P, n=10)
    cos = [(-(factors[4] @ factors[j]) / (np.linalg.norm(factors[4]) * np.linalg.norm(factors[j])), j)
           for j in range(15) if j != 4]
    assert idx.tolist() == [j for _, j in sorted(cos)[:10]]
    assert len(set(idx)) == 10 and 4 not in idx


def test_batched_item_neighbours_match_single_queries():
    rng = np.random.default_rng(2)
    P = ma.product_vectors(rng.normal(size=(11, 4)))
    idx, score = ma.item_neighbors(P, n=10, batch=3)
    for j in range(11):
        assert idx[j].tolist() == ma.recommend_similar(j, P, n=10)[0].tolist()
    assert score.dtype == np.float32


def test_customer_scores_are_weighted_sums_of_item_cosines_and_skip_seen():
    rng = np.random.default_rng(3)
    factors = rng.normal(size=(14, 3))
    P = ma.product_vectors(factors)
    A = sp.csr_matrix(([2.0, 0.5, 1.0], ([0, 0, 0], [1, 6, 9])), shape=(1, 14))
    idx, score = ma.recommend_for_customer(0, A, P, n=10)
    unit = factors / np.linalg.norm(factors, axis=1, keepdims=True)
    brute = {c: 2.0 * unit[1] @ unit[c] + 0.5 * unit[6] @ unit[c] + 1.0 * unit[9] @ unit[c]
             for c in range(14) if c not in (1, 6, 9)}
    expected = sorted(brute, key=lambda c: (-brute[c], c))[:10]
    assert idx.tolist() == expected
    assert score.tolist() == pytest.approx([brute[c] for c in expected])
    assert ma.recommend_all_customers(A, P)[0].tolist() == expected


def test_recommendations_return_fewer_when_catalogue_is_exhausted():
    P = ma.product_vectors(np.eye(4) + 0.1)
    A = sp.csr_matrix(np.array([[1.0, 1.0, 0, 0]]))
    idx, _ = ma.recommend_for_customer(0, A, P, n=10)
    assert sorted(idx[idx >= 0].tolist()) == [2, 3] and (idx[2:] == -1).all()


def test_best_sellers_rank_by_training_quantity_and_refill_after_seen():
    Q = sp.csr_matrix(np.array([
        [100.0, 0, 1, 0, 0],     # product 0 has the most units though only one buyer
        [0, 3, 1, 2, 0],
        [0, 3, 1, 2, 4],
    ]))
    totals = ma.best_seller_totals(Q)
    assert totals.tolist() == [100, 6, 3, 4, 4]
    A = ma.weighted_matrix(Q, "log")
    recs = ma.best_seller_recommendations(A, totals, n=3)
    assert recs[0].tolist() == [1, 3, 4]          # 0 and 2 already bought; 3 beats 4 on index tie
    assert recs[1].tolist() == [0, 4, -1]
    assert recs[2].tolist() == [0, -1, -1]


def test_heldout_targets_and_hit_rate_denominators():
    customer_ids = np.array([10, 20, 30])          # matrix rows 0, 1, 2
    stock_codes = np.array(["A", "B", "C", "D"], dtype=object)
    test = pd.DataFrame({
        "Customer ID": [10, 10, 10, 20, 30, 30, 99, 99],
        "StockCode":   ["B", "C", "C", "Z", "D", "Z", "A", "B"],
    })
    targets, coverage = ma.heldout_targets(test, customer_ids, stock_codes)
    assert {r: t.tolist() for r, t in targets.items()} == {0: [1, 2], 2: [3]}
    assert coverage["total_test_customers"] == 4
    assert coverage["cold_start_customers"] == 1               # customer 99
    assert coverage["warm_customers_no_eligible_product"] == 1  # customer 20 only bought Z
    assert coverage["eligible_customers"] == 2
    assert coverage["total_test_pairs"] == 7                    # duplicate (10, C) line counted once
    assert coverage["eligible_test_pairs"] == 3
    # Three distinct coverage rates: customers (10 and 30 of 4), pairs (3 of 7), distinct products (A-D of A-D, Z).
    assert coverage["eligible_customer_coverage_pct"] == pytest.approx(100 * 2 / 4)
    assert coverage["eligible_pair_coverage_pct"] == pytest.approx(100 * 3 / 7)
    assert coverage["test_products"] == 5 and coverage["test_products_in_catalogue"] == 4
    assert coverage["eligible_product_coverage_pct"] == pytest.approx(100 * 4 / 5)

    recs = np.array([[1, 2, 0], [0, 1, 2], [0, 1, 2]])          # row 0 hits twice, row 2 misses
    result = ma.evaluate(recs, targets, coverage)
    assert result["hits"] == 1 and result["evaluated_customers"] == 2
    assert result["hit_rate_at_10"] == 0.5
    assert result["warm_hit_rate_at_10"] == pytest.approx(1 / 3)
    assert result["all_test_hit_rate_at_10"] == pytest.approx(1 / 4)
    assert result["recall_at_10"] == pytest.approx((1.0 + 0.0) / 2)


def test_recall_denominator_is_not_capped_at_ten():
    # One customer with 20 held-out products, one of which is in the top 10: recall 1/20, not 1/10.
    targets = {0: np.arange(100, 120)}
    coverage = {"warm_test_customers": 1, "total_test_customers": 1}
    recs = np.array([[100, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    result = ma.evaluate(recs, targets, coverage)
    assert result["recall_at_10"] == pytest.approx(0.05)
    assert result["hit_rate_at_10"] == 1.0
    # Two customers: 1/20 and 2/4 average to 0.275; padding (-1) never counts as a hit.
    targets[1] = np.array([0, 1, 2, 3])
    recs = np.vstack([recs, [0, 1, -1, -1, -1, -1, -1, -1, -1, -1]])
    assert ma.evaluate(recs, targets, coverage)["recall_at_10"] == pytest.approx((0.05 + 0.5) / 2)


def test_tiny_catalogue_pads_every_method_to_n():
    # Three products, default n = 10: every method returns 10 columns, -1 / -inf padding.
    P = ma.product_vectors(np.array([[1.0, 0.2], [0.3, 1.0], [0.8, 0.7]]))
    A = sp.csr_matrix(np.array([[1.0, 0, 0], [0, 0, 0], [1.0, 1.0, 1.0], [0, 2.0, 0]]))

    def check(idx, banned):
        assert idx.shape == (ma.TOP_N,)
        real = idx[idx >= 0]
        assert len(set(real.tolist())) == len(real)          # no duplicates
        assert not set(real.tolist()) & set(banned)            # no self / already-bought products
        assert (idx[len(real):] == -1).all()                   # padding only at the end

    for j in range(3):
        idx, score = ma.recommend_similar(j, P)
        check(idx, {j})
        assert len(idx[idx >= 0]) == 2 and np.isneginf(score[2:]).all()
    item_idx, item_score = ma.item_neighbors(P)
    assert item_idx.shape == item_score.shape == (3, ma.TOP_N)
    for j in range(3):
        check(item_idx[j], {j})
    customer_recs = ma.recommend_all_customers(A, P)
    totals = ma.best_seller_totals(A)
    baseline_recs = ma.best_seller_recommendations(A, totals)
    assert customer_recs.shape == baseline_recs.shape == (4, ma.TOP_N)
    for r in range(4):
        seen = set(A[r].indices.tolist())
        check(customer_recs[r], seen)
        check(baseline_recs[r], seen)
        assert ma.recommend_for_customer(r, A, P)[0].tolist() == customer_recs[r].tolist()
    assert (customer_recs[2] == -1).all()                      # bought everything: nothing left


def test_top_n_handles_zero_candidates():
    idx, score = ma.top_n(np.empty((2, 0)), 4)
    assert idx.shape == score.shape == (2, 4) and (idx == -1).all() and np.isneginf(score).all()
    A = sp.csr_matrix((3, 0))
    assert (ma.recommend_all_customers(A, np.empty((0, 2))) == -1).all()
    assert ma.item_neighbors(np.empty((0, 2)))[0].shape == (0, ma.TOP_N)


def test_product_theme_needs_half_the_products():
    assert ma.product_theme(["CHILDRENS TOY", "KIDS BEAKER", "LUNCH BAG", "X", "Y", "Z"]) == "children's ranges"
    assert ma.product_theme(["TEA GLASS", "TEACUP", "PARASOL", "SILK FAN", "X"]) == "mixed gifts"


def test_paired_comparison_counts_discordant_customers():
    a = np.array([1, 1, 1, 0, 0, 1])
    b = np.array([1, 0, 0, 0, 1, 1])
    out = ma.paired_comparison(a, b, n_boot=200)
    assert out["only_first_hit"] == 2 and out["only_second_hit"] == 1
    assert out["difference"] == pytest.approx(1 / 6)
    assert out["ci95_low"] <= out["difference"] <= out["ci95_high"]


def test_component_orientation_and_volume_description():
    rng = np.random.default_rng(4)
    volume = rng.gamma(2.0, 50.0, size=200)
    dense = np.outer(volume, rng.uniform(0.5, 1.5, size=8)).round() + 1
    dense[rng.random(dense.shape) < 0.3] = 0          # varied basket breadth
    dense[:, 0] = np.maximum(dense[:, 0], 1)
    Q = sp.csr_matrix(dense)
    A = ma.weighted_matrix(Q, "log")
    svd = TruncatedSVD(2, algorithm="arpack", random_state=42).fit(A)
    customer, product, _ = ma.factor_arrays(svd, A)
    out = ma.describe_component(-customer, -product, Q, [f"P{j}" for j in range(8)])
    assert out["volume_correlation"] > 0.6 and out["same_sign_loadings_pct"] == 100
    assert out["description"].startswith("Overall buying volume")


def test_clusters_cover_every_customer_with_stable_ids():
    rng = np.random.default_rng(5)
    customer = np.abs(rng.normal(size=(120, 3))) + 0.1
    product = np.abs(rng.normal(size=(9, 3)))
    coords, labels = ma.cluster_customers(customer, product, n_clusters=4)
    assert coords.shape == (120, 2) and np.bincount(labels).sum() == 120
    medians = [np.median(coords[labels == s, 0]) for s in range(4)]
    assert medians == sorted(medians, reverse=True)


def test_mix_lift_ignores_overall_volume():
    # Segment rows buy product 0 and 1 in the same proportions as everyone else, plus product 2.
    binary = sp.csr_matrix(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=float))
    lift = ma.mix_lift(binary, np.array([True, True, False, False]), min_buyers=1)
    assert lift[2] > 1 > lift[0] and lift[0] == pytest.approx(lift[1])


def test_write_json_turns_non_finite_floats_into_null(tmp_path):
    report = {
        "python": [float("nan"), float("inf"), -float("inf"), 0.25],
        "numpy": {"nan": np.float64("nan"), "inf": np.float32("inf"), "ok": np.float32(0.5), "n": np.int64(7)},
        "array": np.array([[1.5, np.nan], [-np.inf, 2.0]]),
        "nested": [{"drop": np.nan, "k": (3, np.float64(0.125))}],
    }
    path = tmp_path / "report.json"
    ma.write_json(report, path)

    def reject(constant):
        raise ValueError(f"invalid JSON constant {constant}")

    parsed = json.loads(path.read_text(), parse_constant=reject)
    assert parsed == {
        "python": [None, None, None, 0.25],
        "numpy": {"nan": None, "inf": None, "ok": 0.5, "n": 7},
        "array": [[1.5, None], [None, 2.0]],
        "nested": [{"drop": None, "k": [3, 0.125]}],
    }


def test_peak_rss_sampler_sees_a_large_temporary_allocation():
    with ma.PeakRSSSampler(interval=0.0005) as sampler:
        baseline = ma._rss()
        block = np.ones(40_000_000 // 8)
        block[::512] += 1                     # touch every page
        import time
        time.sleep(0.02)
        del block
    assert sampler.peak - baseline > 30e6
