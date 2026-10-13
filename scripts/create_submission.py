"""Build the stakeholder memo (J2) and the shared facts file used by the slides (J3).

Every number comes from the aggregate reports written by build_matrix.py and
model_analysis.py. Nothing here is typed in by hand, so rerunning the analysis
and then this script keeps the memo and slides in step with the results.

Run from the project root:

    python scripts/create_submission.py check
        Validate reports/ and print the facts that the memo and slides will use.

    python scripts/create_submission.py facts [--output PATH]
        Write the normalized facts JSON (default tmp/artifacts/submission_facts.json).
        scripts/create_slides.mjs reads this file.

    python scripts/create_submission.py memo [--output PATH] [--replace] [--draft-only]
        Build, check (one page, word count, privacy scan) and render the memo.
        Default output is output/pdf/memo.pdf. If that file already exists the
        validated memo goes to a new output/pdf/memo-<stamp>.pdf instead,
        unless --replace is given, in which case the canonical file is replaced
        by the validated bytes. --draft-only builds, checks and renders a private
        draft without publishing (it does not need the customer map).

Private drafts, renders and validation reports go to tmp/artifacts/memo/.
Needs reportlab and pypdf; rendering uses pdftoppm (set PDFTOPPM to override).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "tmp" / "artifacts"
DEFAULT_FACTS = ARTIFACTS / "submission_facts.json"
DEFAULT_MEMO = ROOT / "output" / "pdf" / "memo.pdf"
CUSTOMER_INDEX = ROOT / "data" / "processed" / "customer_index.csv"
PRODUCT_INDEX = ROOT / "data" / "processed" / "product_index.csv"
BUNDLED_PDFTOPPM = Path(
    "/Users/samopk/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm"
)

FACTS_SCHEMA = "ds5020-submission-facts.v1"
CURVE_KS = [10, 20, 50, 100, 200]
DATASET_CITATION = (
    "Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5CG6D. Licence CC BY 4.0."
)
MEMO_WORD_LIMIT = 500

# Short plain-language labels for the comparison table. Leave empty to use the
# report's own component_1_label or component_1_description. Only fill these in
# after reading the actual descriptions in reports/results.json.
COMPONENT_1_SHORT = {"svd": "", "pca": ""}

# Where each field lives. The first path that exists wins; results.json is the
# primary interface and matrix_report.json (DE5) the documented fallback.
FIELDS = {
    "chosen_k": [("results", "chosen_k")],
    "seed": [("results", "seed")],
    "matrix.customers": [("results", "matrix.customers"), ("matrix", "customers")],
    "matrix.products": [("results", "matrix.products"), ("matrix", "products")],
    "matrix.nonzeros": [("results", "matrix.nonzeros"), ("matrix", "nonzeros")],
    "matrix.sparsity_pct": [("results", "matrix.sparsity_pct"), ("matrix", "sparsity_pct")],
    "matrix.dense_mb": [("results", "matrix.dense_mb"), ("matrix", "dense_mb")],
    "matrix.sparse_mb": [("results", "matrix.sparse_mb"), ("matrix", "sparse_mb")],
    "matrix.min_buyers": [
        ("results", "matrix.min_buyers_per_product"),
        ("matrix", "min_buyers_per_product"),
    ],
    "data.raw_rows": [("results", "data.raw_rows"), ("matrix", "raw_rows")],
    "data.clean_rows": [("results", "data.clean_rows"), ("matrix", "clean_rows")],
    "data.train_rows": [("results", "data.train_rows"), ("matrix", "train_rows")],
    "data.test_rows": [("results", "data.test_rows"), ("matrix", "test_rows")],
    "data.excluded_repeat_rows": [
        ("results", "data.excluded_repeat_rows"),
        ("matrix", "excluded_repeat_rows"),
    ],
    "data.cutoff": [("results", "data.cutoff"), ("matrix", "cutoff")],
    "data.train_start": [("results", "data.train_start"), ("matrix", "train_start")],
    "data.train_end": [("results", "data.train_end"), ("matrix", "train_end")],
    "data.test_start": [("results", "data.test_start"), ("matrix", "test_start")],
    "data.test_end": [("results", "data.test_end"), ("matrix", "test_end")],
    "evaluation.eligible_customers": [("results", "evaluation.eligible_customers")],
    "evaluation.total_test_customers": [("results", "evaluation.total_test_customers")],
    "evaluation.cold_start_customers": [("results", "evaluation.cold_start_customers")],
    "evaluation.eligible_customer_coverage_pct": [
        ("results", "evaluation.eligible_customer_coverage_pct")
    ],
    "evaluation.eligible_test_pairs": [("results", "evaluation.eligible_test_pairs")],
    "evaluation.total_test_pairs": [("results", "evaluation.total_test_pairs")],
    "evaluation.test_products": [("results", "evaluation.test_products")],
    "evaluation.test_products_in_catalogue": [("results", "evaluation.test_products_in_catalogue")],
    "baseline.hit_rate_at_10": [("results", "baseline.hit_rate_at_10")],
    "baseline.hits": [("results", "baseline.hits")],
    "baseline.evaluated_customers": [("results", "baseline.evaluated_customers")],
    "nightly.fit_seconds": [("results", "nightly.fit_seconds")],
    "nightly.normalization_seconds": [("results", "nightly.normalization_seconds")],
    "nightly.item_scoring_seconds": [("results", "nightly.item_scoring_seconds")],
    "nightly.total_seconds": [("results", "nightly.total_seconds")],
}
for _model in ("svd", "pca"):
    for _name in (
        "fit_seconds",
        "peak_rss_mb",
        "rss_increase_mb",
        "factor_mb",
        "hit_rate_at_10",
        "hits",
        "evaluated_customers",
        "component_1_description",
        "component_1_volume_correlation",
    ):
        FIELDS[f"{_model}.{_name}"] = [("results", f"{_model}.{_name}")]
FIELDS["svd.relative_error"] = [("results", "svd.relative_error")]
FIELDS["pca.variance_explained"] = [("results", "pca.variance_explained")]

OPTIONAL_FIELDS = {
    "evaluation.warm_no_eligible_customers": [
        ("results", "evaluation.warm_customers_no_eligible_product"),
        ("results", "evaluation.warm_no_eligible_customers"),
        ("results", "evaluation.out_of_catalogue_customers"),
    ],
    "nightly.customer_scoring_seconds": [("results", "nightly.customer_scoring_seconds")],
    "svd.component_1_label": [("results", "svd.component_1_label")],
    "pca.component_1_label": [("results", "pca.component_1_label")],
    "pca.relative_error": [("results", "pca.relative_error")],
    "products_dropped_by_min_buyers": [("matrix", "products_dropped_by_min_buyers")],
    "train_products": [("matrix", "train_products")],
}


class InputError(RuntimeError):
    """Raised when reports are missing, incomplete or inconsistent."""


# --------------------------------------------------------------------------
# Loading


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def dig(obj, dotted: str):
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise InputError(f"{rel(path)} is not a PNG file")
    return struct.unpack(">II", header[16:24])


def load_inputs(reports: Path, require_map: bool = True) -> dict:
    required = {
        "results": reports / "results.json",
        "comparison": reports / "comparison.csv",
        "svd_curve": reports / "svd_curve.csv",
        "segments": reports / "segments.csv",
    }
    if require_map:
        # model_analysis.py writes figures/ at the project root; reports/figures/ is the older plan.
        candidates = [ROOT / "figures" / "customer_map.png", reports / "figures" / "customer_map.png"]
        required["customer_map"] = next((c for c in candidates if c.is_file()), candidates[0])
    optional = {
        "matrix": reports / "matrix_report.json",
        "sensitivity_csv": reports / "sensitivity.csv",
        "segment_profiles": reports / "segment_profiles.json",
    }
    missing = [rel(p) for p in required.values() if not p.is_file() or p.stat().st_size == 0]
    if require_map and "figures/customer_map.png" in missing:
        missing[missing.index("figures/customer_map.png")] = "figures/customer_map.png (or reports/figures/)"
    if missing:
        raise InputError(
            "Reports needed for the memo and slides are not ready yet:\n  "
            + "\n  ".join(missing)
            + "\nRun `python build_matrix.py` and `python model_analysis.py` from the project root first."
        )
    paths = dict(required)
    paths.update({k: p for k, p in optional.items() if p.is_file()})
    try:
        loaded = {
            "results": json.loads(paths["results"].read_text(encoding="utf-8")),
            "matrix": json.loads(paths["matrix"].read_text(encoding="utf-8")) if "matrix" in paths else {},
            "comparison": read_csv(paths["comparison"]),
            "svd_curve": read_csv(paths["svd_curve"]),
            "segments": read_csv(paths["segments"]),
            "segment_profiles": (json.loads(paths["segment_profiles"].read_text(encoding="utf-8"))
                                 if "segment_profiles" in paths else {}),
        }
    except (json.JSONDecodeError, UnicodeDecodeError, csv.Error) as error:
        raise InputError(f"Could not parse a report: {error}") from error
    loaded["paths"] = paths
    return loaded


def resolve_fields(inputs: dict) -> tuple[dict, dict]:
    values, sources, missing = {}, {}, []
    for name, candidates in {**FIELDS, **OPTIONAL_FIELDS}.items():
        for source, dotted in candidates:
            value = dig(inputs.get(source, {}), dotted)
            if value is not None:
                values[name] = value
                sources[name] = f"{rel(inputs['paths'][source])}:{dotted}"
                break
        else:
            if name in FIELDS:
                missing.append(f"{name} (looked for {', '.join(f'{s}:{d}' for s, d in candidates)})")
    if missing:
        raise InputError("reports/results.json is missing required fields:\n  " + "\n  ".join(missing))
    return values, sources


# --------------------------------------------------------------------------
# Validation helpers


def number(values: dict, name: str, low=None, high=None, integer=False) -> float:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InputError(f"{name} must be a finite number, got {value!r}")
    if integer and float(value) != int(value):
        raise InputError(f"{name} must be a whole number, got {value!r}")
    if low is not None and value < low or high is not None and value > high:
        raise InputError(f"{name}={value!r} is outside [{low}, {high}]")
    return int(value) if integer else float(value)


def text(values: dict, name: str) -> str:
    value = values[name]
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{name} must be a nonempty string")
    return value.strip()


def timestamp(value: str, name: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", ""))
    except ValueError as error:
        raise InputError(f"{name} is not an ISO date/time: {value!r}") from error


def check_rate(rate: float, hits: int, n: int, label: str) -> None:
    if n <= 0 or not 0 <= hits <= n:
        raise InputError(f"{label}: hits={hits} evaluated={n} is not a valid count")
    if abs(hits / n - rate) > 5e-4:
        raise InputError(f"{label}: hit_rate_at_10={rate} disagrees with hits/evaluated={hits / n:.6f}")


def parse_list(cell: str) -> list[str]:
    cell = (cell or "").strip()
    if not cell:
        return []
    if cell.startswith("["):
        try:
            items = json.loads(cell)
        except json.JSONDecodeError as error:
            raise InputError(f"Could not parse list cell {cell[:60]!r}") from error
        return [str(i).strip() for i in items if str(i).strip()]
    for separator in (";", "|", "\n"):
        if separator in cell:
            return [part.strip() for part in cell.split(separator) if part.strip()]
    return [cell]


# Trailing annotations such as "(53%)", "(lift 5.4)" or "(mix lift 1.7)".
ANNOTATION = re.compile(r"\s*\((?:(?P<label>[a-z][a-z ]*?)\s+)?(?P<value>[0-9]+(?:\.[0-9]+)?)%?\)\s*$", re.I)


def strip_annotation(item: str) -> str:
    return ANNOTATION.sub("", item).strip()


def annotation_value(item: str, label: str) -> float | None:
    """Value of an annotation whose label ends with `label`, e.g. lift in "(mix lift 1.7)"."""
    match = ANNOTATION.search(item)
    if match and (match.group("label") or "").lower().split()[-1:] == [label]:
        return float(match.group("value"))
    return None


def optional_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


# One example product per group, chosen to illustrate the group's named theme. Each entry
# maps a phrase in the group name to preferred product phrases; the product must appear in
# the group's actual distinctive or top-product lists, otherwise the build stops.
SEGMENT_EXAMPLES = [
    ("High-volume wholesalers", ["DOLLY GIRL BEAKER", "SPACEBOY MINI BACKPACK"]),
    ("home decor", ["WHITE PEARL BEADED HEART"]),
    ("Smaller regular buyers", ["NURSE'S BAG SOFT TOY"]),
    ("Occasional small buyers", ["ANTIQUE SILVER TEA GLASS", "EDWARDIAN PARASOL"]),
]


def example_product(segment: dict) -> str:
    candidates = segment["distinctive_products"] + segment["top_products"]
    for name_phrase, product_phrases in SEGMENT_EXAMPLES:
        if name_phrase.lower() in segment["name"].lower():
            for phrase in product_phrases:
                match = next((p for p in candidates if phrase.lower() in p.lower()), None)
                if match:
                    return match
            raise InputError(f"None of {product_phrases} is in the product lists of segment {segment['name']!r}; "
                             "update SEGMENT_EXAMPLES in scripts/create_submission.py from the actual lists")
    raise InputError(f"No example rule for segment {segment['name']!r}; add one to SEGMENT_EXAMPLES "
                     "using a product from its distinctive list")


def find_column(rows: list[dict], *candidates: str, required=True) -> str | None:
    columns = list(rows[0].keys()) if rows else []
    lowered = {c.strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    if required:
        raise InputError(f"Expected one of the columns {candidates} but found {columns}")
    return None


def uk_share(countries_cell: str) -> float | None:
    """Share of a segment's customers in the UK, as a fraction, if the report states it."""
    match = re.search(r"United Kingdom\D{0,6}([0-9]+(?:\.[0-9]+)?)\s*(%)?", countries_cell or "")
    if not match:
        return None
    value = float(match.group(1))
    return value / 100 if match.group(2) or value > 1 else value


# --------------------------------------------------------------------------
# Formatting (shared by the memo and the slides through the facts file)


def fmt_int(value: int) -> str:
    return f"{int(value):,}"


def fmt_rate(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def fmt_pct(pct: float, places=1) -> str:
    return f"{pct:.{places}f}%"


def fmt_mb(mb: float) -> str:
    return f"{mb:.2f} MB" if mb < 10 else f"{mb:.1f} MB"


def fmt_seconds(seconds: float) -> str:
    if seconds < 0.01:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 1:
        return f"{seconds:.3f} s"
    return f"{seconds:.2f} s" if seconds < 10 else f"{seconds:.1f} s"


def fmt_date(moment: datetime) -> str:
    return f"{moment.day} {moment.strftime('%b %Y')}"


def sentence_case(description: str) -> str:
    """Product descriptions are upper case in the source data."""
    lowered = description.strip().lower()
    return lowered[:1].upper() + lowered[1:]


WEIGHTING_LABELS = {
    "count": "Raw count",
    "raw": "Raw count",
    "raw_count": "Raw count",
    "quantity": "Raw count",
    "binary": "Bought or not (1/0)",
    "log": "log(1 + quantity)",
    "log1p": "log(1 + quantity)",
}


# --------------------------------------------------------------------------
# Facts


def build_facts(reports: Path = REPORTS, require_map: bool = True) -> dict:
    inputs = load_inputs(reports, require_map)
    values, sources = resolve_fields(inputs)
    results = inputs["results"]
    warnings: list[str] = []

    k = number(values, "chosen_k", 1, integer=True)
    customers = number(values, "matrix.customers", 1, integer=True)
    products = number(values, "matrix.products", 1, integer=True)
    nonzeros = number(values, "matrix.nonzeros", 1, integer=True)
    sparsity = number(values, "matrix.sparsity_pct", 0, 100)
    dense_mb = number(values, "matrix.dense_mb", 0)
    sparse_mb = number(values, "matrix.sparse_mb", 0)
    if abs(dense_mb - customers * products * 8 / 1e6) > 0.01 * dense_mb:
        raise InputError("matrix.dense_mb does not match customers x products x 8 bytes")
    if abs(sparsity - 100 * (1 - nonzeros / (customers * products))) > 0.01:
        raise InputError("matrix.sparsity_pct does not match nonzeros / cells")

    data = {
        key: number(values, f"data.{key}", 0, integer=True)
        for key in ("raw_rows", "clean_rows", "train_rows", "test_rows", "excluded_repeat_rows")
    }
    dates = {
        key: timestamp(values[f"data.{key}"], f"data.{key}")
        for key in ("cutoff", "train_start", "train_end", "test_start", "test_end")
    }
    if not dates["train_start"] <= dates["train_end"] < dates["cutoff"] <= dates["test_start"] <= dates["test_end"]:
        raise InputError("Training and test dates are not in chronological order around the cutoff")

    evaluation = {
        key: number(values, f"evaluation.{key}", 0, integer=True)
        for key in ("eligible_customers", "total_test_customers", "cold_start_customers",
                    "eligible_test_pairs", "total_test_pairs")
    }
    evaluation["eligible_customer_coverage_pct"] = number(values, "evaluation.eligible_customer_coverage_pct", 0, 100)
    # Three different coverage measures; keep them apart. Percentages come from the counts.
    for key in ("test_products", "test_products_in_catalogue"):
        evaluation[key] = number(values, f"evaluation.{key}", 0, integer=True)
    if not 0 < evaluation["test_products_in_catalogue"] <= evaluation["test_products"]:
        raise InputError("test_products_in_catalogue must be between 1 and test_products")
    if not 0 < evaluation["eligible_test_pairs"] <= evaluation["total_test_pairs"]:
        raise InputError("eligible_test_pairs must be between 1 and total_test_pairs")
    evaluation["product_coverage_pct"] = 100 * evaluation["test_products_in_catalogue"] / evaluation["test_products"]
    evaluation["pair_coverage_pct"] = 100 * evaluation["eligible_test_pairs"] / evaluation["total_test_pairs"]
    for field, derived in (("eligible_pair_coverage_pct", evaluation["pair_coverage_pct"]),
                           ("eligible_product_coverage_pct", evaluation["product_coverage_pct"])):
        reported = dig(results, f"evaluation.{field}")
        if not isinstance(reported, (int, float)) or abs(reported - derived) > 0.05:
            raise InputError(f"evaluation.{field}={reported!r} disagrees with its counts ({derived:.2f}%)")
    if evaluation["eligible_customers"] + evaluation["cold_start_customers"] > evaluation["total_test_customers"]:
        raise InputError("Eligible plus cold-start customers exceed all test customers")
    if abs(100 * evaluation["eligible_customers"] / evaluation["total_test_customers"]
           - evaluation["eligible_customer_coverage_pct"]) > 0.05:
        raise InputError("eligible_customer_coverage_pct disagrees with eligible/total test customers")
    if "evaluation.warm_no_eligible_customers" in values:
        evaluation["warm_no_eligible_customers"] = number(
            values, "evaluation.warm_no_eligible_customers", 0, integer=True)

    models = {}
    for model in ("svd", "pca", "baseline"):
        rate = number(values, f"{model}.hit_rate_at_10", 0, 1)
        hits = number(values, f"{model}.hits", 0, integer=True)
        n = number(values, f"{model}.evaluated_customers", 1, integer=True)
        check_rate(rate, hits, n, model)
        models[model] = {"hit_rate_at_10": rate, "hits": hits, "evaluated_customers": n}
    cohort = {m["evaluated_customers"] for m in models.values()}
    if cohort != {evaluation["eligible_customers"]}:
        raise InputError(
            f"SVD, PCA and baseline must use the same primary cohort of eligible customers; got {sorted(cohort)}"
            f" vs eligible_customers={evaluation['eligible_customers']}")
    for model in ("svd", "pca"):
        m = models[model]
        m["fit_seconds"] = number(values, f"{model}.fit_seconds", 0)
        m["peak_rss_mb"] = number(values, f"{model}.peak_rss_mb", 0)
        m["rss_increase_mb"] = number(values, f"{model}.rss_increase_mb")
        m["factor_mb"] = number(values, f"{model}.factor_mb", 0)
        m["component_1_description"] = text(values, f"{model}.component_1_description")
        m["component_1_volume_correlation"] = number(values, f"{model}.component_1_volume_correlation", -1, 1)
        label = COMPONENT_1_SHORT.get(model) or values.get(f"{model}.component_1_label")
        if not label:
            # Descriptions read "Headline: detail (r = ...)"; keep the headline and the measured r.
            description = m["component_1_description"]
            headline = description.split(":", 1)[0].strip() if ":" in description else description
            if len(headline) > 40:
                raise InputError(
                    f"{model}.component_1_description is too long for the slide table. Read it in "
                    f"reports/results.json and set COMPONENT_1_SHORT['{model}'] in scripts/create_submission.py "
                    "to a short faithful label.")
            label = f"{headline} (r = {m['component_1_volume_correlation']:.2f})"
        m["component_1_short"] = label.strip()
        for bound in ("hit_rate_ci95_low", "hit_rate_ci95_high"):
            if isinstance(dig(results, f"{model}.{bound}"), (int, float)):
                m[bound] = float(dig(results, f"{model}.{bound}"))
    svd, pca, base = models["svd"], models["pca"], models["baseline"]
    svd["relative_error"] = number(values, "svd.relative_error", 0, 1)
    pca["variance_explained"] = number(values, "pca.variance_explained", 0, 1)
    if "pca.relative_error" in values:
        pca["relative_error"] = number(values, "pca.relative_error", 0, 1)

    expected_factor = (customers + products) * k * 8 / 1e6
    if abs(svd["factor_mb"] - expected_factor) > 0.02 * expected_factor:
        raise InputError(
            f"svd.factor_mb={svd['factor_mb']} should cover both customer and product factors "
            f"(about {expected_factor:.3f} MB at k={k})")
    mean_mb = products * 8 / 1e6
    if pca["factor_mb"] - svd["factor_mb"] < 0.5 * mean_mb:
        raise InputError(
            f"pca.factor_mb={pca['factor_mb']} does not appear to include the {mean_mb:.4f} MB column-mean vector")

    nightly = {key: number(values, f"nightly.{key}", 0) for key in
               ("fit_seconds", "normalization_seconds", "item_scoring_seconds", "total_seconds")}
    parts = nightly["fit_seconds"] + nightly["normalization_seconds"] + nightly["item_scoring_seconds"]
    if abs(parts - nightly["total_seconds"]) > max(0.05, 0.02 * nightly["total_seconds"]):
        warnings.append(f"nightly.total_seconds={nightly['total_seconds']} differs from its parts ({parts:.3f} s)")
    if "nightly.customer_scoring_seconds" in values:
        nightly["customer_scoring_seconds"] = number(values, "nightly.customer_scoring_seconds", 0)

    # ML2 curve from svd_curve.csv
    curve_rows = inputs["svd_curve"]
    k_col = find_column(curve_rows, "k", "n_components", "components")
    err_col = find_column(curve_rows, "relative_error", "relative_reconstruction_error", "error")
    curve = sorted(
        ({"k": int(float(r[k_col])), "relative_error": float(r[err_col])} for r in curve_rows),
        key=lambda row: row["k"])
    if [row["k"] for row in curve] != CURVE_KS:
        raise InputError(f"svd_curve.csv must contain k = {CURVE_KS}; found {[row['k'] for row in curve]}")
    if any(not 0 < row["relative_error"] < 1 for row in curve):
        raise InputError("svd_curve.csv relative errors must be fractions between 0 and 1")
    chosen = next((row for row in curve if row["k"] == k), None)
    if chosen is None or abs(chosen["relative_error"] - svd["relative_error"]) > 1e-3:
        raise InputError("svd_curve.csv error at the chosen k disagrees with svd.relative_error")

    # ML7 sensitivity
    sensitivity = []
    for entry in results.get("sensitivity") or []:
        weighting = str(entry.get("weighting", "")).strip()
        rate = entry.get("hit_rate_at_10")
        if not weighting or not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
            raise InputError(f"Bad sensitivity entry {entry!r}")
        label = WEIGHTING_LABELS.get(weighting.lower())
        if label is None:
            raise InputError(f"Unknown sensitivity weighting {weighting!r}; add a label to WEIGHTING_LABELS")
        row = {"weighting": weighting, "label": label, "hit_rate_at_10": float(rate)}
        if "hits" in entry and "evaluated_customers" in entry:
            check_rate(float(rate), int(entry["hits"]), int(entry["evaluated_customers"]), f"sensitivity {weighting}")
            row.update(hits=int(entry["hits"]), evaluated_customers=int(entry["evaluated_customers"]))
        sensitivity.append(row)
    if not sensitivity:
        raise InputError("results.json has no sensitivity entries (ML7 count/binary/log)")

    # ML6 segments from segments.csv
    seg_rows = inputs["segments"]
    name_col = find_column(seg_rows, "name", "segment_name")
    count_col = find_column(seg_rows, "customers", "count", "n_customers")
    products_col = find_column(seg_rows, "top_products", "products")
    countries_col = find_column(seg_rows, "countries", "top_countries")
    distinct_col = find_column(seg_rows, "distinctive_products", "lift_products", required=False)
    id_col = find_column(seg_rows, "segment_id", "segment", "cluster", required=False)
    listed = {str(s.get("name", "")).strip(): s for s in results.get("segments") or [] if isinstance(s, dict)}
    segments = []
    for row in seg_rows:
        name = row[name_col].strip()
        count = int(float(row[count_col]))
        # The CSV annotates items, e.g. "HEART OF WICKER SMALL (53%)" or "... (lift 5.4)".
        top = [strip_annotation(p) for p in parse_list(row[products_col])]
        distinctive_cells = parse_list(row[distinct_col]) if distinct_col else []
        distinctive = [strip_annotation(p) for p in distinctive_cells]
        lifts = [annotation_value(p, "lift") for p in distinctive_cells]
        structured = listed.get(name, {})
        if structured:
            if int(structured.get("customers", count)) != count:
                raise InputError(f"Segment {name!r} has different sizes in results.json and segments.csv")
            top = [p.strip() for p in structured.get("top_products") or top]
            distinctive = [p.strip() for p in structured.get("distinctive_products") or distinctive]
        if not name or not top:
            raise InputError(f"segments.csv row {row.get(id_col, '?')} needs a name and top products")
        countries = structured.get("countries")
        if isinstance(countries, list) and all(isinstance(c, dict) for c in countries):
            country_shares = [{"country": str(c["country"]), "share": float(c["customer_share_pct"]) / 100}
                              for c in countries]
        else:
            country_shares = []
        uk = next((c["share"] for c in country_shares if c["country"] == "United Kingdom"), None)
        segments.append({
            "segment_id": row.get(id_col, "") if id_col else "",
            "name": name,
            "customers": count,
            "share": count / customers,
            "top_products": top,
            "distinctive_products": distinctive,
            "distinctive_lifts": lifts,
            "countries": row[countries_col].strip(),
            "country_shares": country_shares,
            "uk_share": uk if uk is not None else uk_share(row[countries_col]),
            "avg_quantity_per_customer": optional_float(row.get("avg_quantity_per_customer")),
            "avg_invoices": optional_float(row.get("avg_invoices")),
            "theme": structured.get("theme"),
        })
    if len(segments) != 4:
        warnings.append(f"Expected four segments, found {len(segments)}")
    if not 3 <= len(segments) <= 6:
        raise InputError("ML6 needs 3 to 6 customer groups")
    if sum(s["customers"] for s in segments) != customers:
        raise InputError(
            f"Segment sizes sum to {sum(s['customers'] for s in segments)} but the matrix has {customers} customers")
    if listed and sorted(listed) != sorted(s["name"] for s in segments):
        raise InputError("Segment names in results.json and segments.csv disagree")
    for segment in segments:
        segment["example_product"] = example_product(segment)

    # ML5 comparison table must exist with both methods
    comparison = inputs["comparison"]
    if len(comparison) < 8 or len(comparison[0]) < 3:
        raise InputError("comparison.csv should hold the eight ML5 measures for PCA and SVD")

    customer_map = inputs["paths"].get("customer_map")
    map_display = results.get("customer_map") if isinstance(results.get("customer_map"), dict) else {}
    if not map_display and customer_map is not None:
        map_display = verified_map_display(customers)
    if customer_map is None:
        warnings.append("customer_map.png not checked (memo draft only)")

    hardware = results.get("hardware") or results.get("environment")
    machine = describe_machine(hardware)
    if machine is None:
        raise InputError("results.json needs a hardware/environment description for the timing claims")

    # Paired differences between methods on the same cohort, when the analysis reports them.
    comparisons = {}
    for key, entry in (dig(results, "evaluation.comparisons") or {}).items():
        if isinstance(entry, dict) and all(isinstance(entry.get(f), (int, float))
                                           for f in ("difference", "ci95_low", "ci95_high")):
            comparisons[key] = {f: float(entry[f]) for f in
                                ("difference", "ci95_low", "ci95_high", "mcnemar_exact_p") if f in entry}
    for key, (first, second) in {"svd_vs_baseline": (svd, base), "svd_vs_pca": (svd, pca)}.items():
        if key in comparisons and abs(comparisons[key]["difference"]
                                      - (first["hit_rate_at_10"] - second["hit_rate_at_10"])) > 1e-6:
            raise InputError(f"evaluation.comparisons.{key}.difference disagrees with the hit rates")

    environment = hardware if isinstance(hardware, dict) else {}
    methodology = {key: environment[key] for key in ("timing", "memory_sampling") if isinstance(
        environment.get(key), str)}
    methodology.update({key: results[key] for key in ("methodology", "benchmark") if key in results})
    if isinstance(inputs["segment_profiles"].get("method"), str):
        methodology["segmentation"] = inputs["segment_profiles"]["method"]

    uk = [s["uk_share"] for s in segments]
    fmt = {
        "k": str(k),
        "customers": fmt_int(customers),
        "products": fmt_int(products),
        "nonzeros": fmt_int(nonzeros),
        "sparsity": fmt_pct(sparsity),
        "dense_mb": fmt_mb(dense_mb),
        "sparse_mb": fmt_mb(sparse_mb),
        "dense_to_factor_ratio": f"{dense_mb / svd['factor_mb']:.0f}",
        "svd_factor_mb": fmt_mb(svd["factor_mb"]),
        "pca_factor_mb": fmt_mb(pca["factor_mb"]),
        "min_buyers": fmt_int(number(values, "matrix.min_buyers", 1, integer=True)),
        **{key: fmt_int(value) for key, value in data.items()},
        **{key: fmt_date(value) for key, value in dates.items()},
        **{key: fmt_int(value) for key, value in evaluation.items() if not key.endswith("_pct")},
        "eligible_customer_coverage_pct": fmt_pct(evaluation["eligible_customer_coverage_pct"]),
        "product_coverage_pct": fmt_pct(evaluation["product_coverage_pct"]),
        "pair_coverage_pct": fmt_pct(evaluation["pair_coverage_pct"]),
        "svd_hit": fmt_rate(svd["hit_rate_at_10"]),
        "pca_hit": fmt_rate(pca["hit_rate_at_10"]),
        "baseline_hit": fmt_rate(base["hit_rate_at_10"]),
        "svd_hits": fmt_int(svd["hits"]),
        "pca_hits": fmt_int(pca["hits"]),
        "baseline_hits": fmt_int(base["hits"]),
        "svd_error": fmt_rate(svd["relative_error"]),
        "pca_variance": fmt_rate(pca["variance_explained"]),
        "svd_fit": fmt_seconds(svd["fit_seconds"]),
        "pca_fit": fmt_seconds(pca["fit_seconds"]),
        "svd_peak": f"{svd['peak_rss_mb']:.0f} MB",
        "pca_peak": f"{pca['peak_rss_mb']:.0f} MB",
        "svd_rss_increase": f"{svd['rss_increase_mb']:.0f} MB",
        "pca_rss_increase": f"{pca['rss_increase_mb']:.0f} MB",
        "svd_corr": f"{svd['component_1_volume_correlation']:.2f}",
        "pca_corr": f"{pca['component_1_volume_correlation']:.2f}",
        "nightly_total": fmt_seconds(nightly["total_seconds"]),
        "nightly_fit": fmt_seconds(nightly["fit_seconds"]),
        "nightly_normalization": fmt_seconds(nightly["normalization_seconds"]),
        "nightly_item_scoring": fmt_seconds(nightly["item_scoring_seconds"]),
        "machine": machine,
        "uk_min": fmt_rate(min(uk)) if all(u is not None for u in uk) else "",
        "uk_max": fmt_rate(max(uk)) if all(u is not None for u in uk) else "",
    }
    if "customer_scoring_seconds" in nightly:
        fmt["nightly_customer_scoring"] = fmt_seconds(nightly["customer_scoring_seconds"])
    if "relative_error" in pca:
        fmt["pca_error"] = fmt_rate(pca["relative_error"])
    for row in curve:
        fmt[f"curve_{row['k']}"] = fmt_rate(row["relative_error"])
    for row in sensitivity:
        fmt[f"sensitivity_{row['weighting'].lower()}"] = fmt_rate(row["hit_rate_at_10"])
    for model in ("svd", "pca", "baseline"):
        if "hit_rate_ci95_low" in models[model]:
            fmt[f"{model}_ci"] = (f"{models[model]['hit_rate_ci95_low'] * 100:.1f}% to "
                                  f"{models[model]['hit_rate_ci95_high'] * 100:.1f}%")
    for key, entry in comparisons.items():
        fmt[f"{key}_diff"] = f"{entry['difference'] * 100:.1f}"
        fmt[f"{key}_ci"] = f"{entry['ci95_low'] * 100:.1f} to {entry['ci95_high'] * 100:.1f}"
        if "mcnemar_exact_p" in entry:
            fmt[f"{key}_p"] = f"{entry['mcnemar_exact_p']:.2g}"

    facts = {
        "schema": FACTS_SCHEMA,
        "results_schema_version": results.get("schema_version"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "chosen_k": k,
        "seed": number(values, "seed", integer=True),
        "matrix": {"customers": customers, "products": products, "nonzeros": nonzeros,
                   "sparsity_pct": sparsity, "dense_mb": dense_mb, "sparse_mb": sparse_mb,
                   "min_buyers": number(values, "matrix.min_buyers", 1, integer=True)},
        "data": {**data, **{key: value.isoformat(sep=" ") for key, value in dates.items()}},
        "evaluation": evaluation,
        "svd": svd,
        "pca": pca,
        "baseline": base,
        "nightly": nightly,
        "curve": curve,
        "sensitivity": sensitivity,
        "segments": segments,
        "comparison_csv": comparison,
        "hardware": hardware,
        # Passed through verbatim so the speaker notes can describe the real method.
        "methodology": methodology,
        "comparisons": comparisons,
        "k_choice_reason": dig(results, "k_choice.reason"),
        "targets": results.get("targets") if isinstance(results.get("targets"), dict) else {},
        "customer_map_display": map_display,
        "customer_map": None if customer_map is None else {
            "path": rel(customer_map), **dict(zip(("width", "height"), png_size(customer_map))),
            "sha256": sha256(customer_map)},
        "outcomes": {
            "svd_beats_baseline": svd["hits"] > base["hits"],
            "svd_beats_pca": svd["hits"] > pca["hits"],
            "svd_vs_baseline_points": 100 * (svd["hit_rate_at_10"] - base["hit_rate_at_10"]),
            "svd_vs_pca_points": 100 * (svd["hit_rate_at_10"] - pca["hit_rate_at_10"]),
            "factors_exceed_csr": svd["factor_mb"] > sparse_mb,
            "curve_still_falling_after_k": any(r["k"] > k and r["relative_error"] < svd["relative_error"]
                                               for r in curve),
            "uk_dominates_every_segment": all(u is not None and u >= 0.5 for u in uk),
            # A paired 95% interval that contains zero means the test cannot separate the two methods.
            "svd_pca_difference_not_established": ("svd_vs_pca" in comparisons
                             and comparisons["svd_vs_pca"]["ci95_low"] <= 0 <= comparisons["svd_vs_pca"]["ci95_high"]),
            "svd_baseline_ci_above_zero": ("svd_vs_baseline" in comparisons
                                           and comparisons["svd_vs_baseline"]["ci95_low"] > 0),
            "best_sensitivity_weighting": max(sensitivity, key=lambda r: r["hit_rate_at_10"])["weighting"],
        },
        "fmt": fmt,
        "citation": DATASET_CITATION,
        "sources": sources,
        "inputs": {key: {"path": rel(path), "sha256": sha256(path)} for key, path in inputs["paths"].items()},
        "warnings": warnings,
    }
    return facts


# How model_analysis.py draws figures/customer_map.png, read from its source. The check
# below stops the build if those lines change, so the slide text cannot drift from the figure.
MAP_SOURCE_MARKERS = [
    "size = np.log10(np.hypot(c1, c2))",
    "mix = np.degrees(np.arctan2(c2, c1))",
    "z = StandardScaler().fit_transform(coords)",
    "_full_extent(ax, coords[:, 0], coords[:, 1])",
]


def verified_map_display(customers: int) -> dict:
    source = ROOT / "model_analysis.py"
    code = source.read_text(encoding="utf-8") if source.is_file() else ""
    missing = [marker for marker in MAP_SOURCE_MARKERS if marker not in code]
    if missing:
        raise InputError("model_analysis.py no longer draws the customer map as expected "
                         f"(missing {missing}); update the map description in scripts/create_submission.py")
    return {
        "description": ("The map plots each customer's size, the log10 distance from the origin on components "
                        "one and two, against mix, the angle between them in degrees. "
                        f"All {customers:,} customers are visible."),
        "alt": (f"Customer map of all {customers:,} anonymous customers, coloured by group: size (log10 distance "
                "from the origin on SVD components 1 and 2) across, product mix (angle in degrees) up and down."),
        "source": "model_analysis.py map_coordinates and plot_customer_map",
    }


def describe_machine(hardware) -> str | None:
    if isinstance(hardware, str) and hardware.strip():
        return hardware.strip()
    if not isinstance(hardware, dict):
        return None
    for key in ("description", "summary", "machine_description"):
        if isinstance(hardware.get(key), str) and hardware[key].strip():
            return hardware[key].strip()
    cpu = str(hardware.get("cpu") or hardware.get("processor") or "").strip()
    if not cpu:
        return None
    platform = str(hardware.get("platform", ""))
    computer = f"an {cpu} Mac" if platform.startswith("macOS") else f"a computer with an {cpu} processor"
    if not cpu[:1].upper() in "AEIOU":
        computer = computer.replace("an ", "a ", 1)
    cores = hardware.get("logical_cpus")
    return f"{computer} ({cores} logical CPUs)" if isinstance(cores, int) else computer


def write_facts(facts: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    # allow_nan=False: the slides parse this with JSON.parse, which rejects NaN.
    temp.write_text(json.dumps(facts, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(output)


# --------------------------------------------------------------------------
# Privacy scan (R4)


def private_ids() -> tuple[set[str], set[str], str]:
    """Customer IDs from the private index, minus stock codes that look the same."""
    if not CUSTOMER_INDEX.is_file():
        return set(), set(), "pattern"
    rows = read_csv(CUSTOMER_INDEX)
    column = next((c for c in rows[0] if "customer" in c.lower()), list(rows[0])[0]) if rows else None
    ids = {str(int(float(r[column]))) for r in rows if column and r[column].strip()}
    codes = set()
    if PRODUCT_INDEX.is_file():
        for r in read_csv(PRODUCT_INDEX):
            codes.update(v.strip().upper() for v in r.values() if v)
    return ids, codes, "index"


def privacy_findings(text_value: str) -> list[str]:
    """Return positions of anything that looks like a customer ID. Never returns the IDs."""
    ids, codes, mode = private_ids()
    findings = []
    for match in re.finditer(r"(?<![\w.,])(\d{5})(?:\.0)?(?![\w,])", text_value):
        token = match.group(1)
        if token in codes:
            continue
        if (mode == "index" and token in ids) or (mode == "pattern" and 12000 <= int(token) <= 18999):
            findings.append(f"possible customer ID at character {match.start()}")
    if re.search(r"customer[ _]?id\s*[:=]?\s*\d", text_value, re.I):
        findings.append("text labels a customer ID value")
    return findings


# --------------------------------------------------------------------------
# Memo


def memo_blocks(facts: dict) -> list[tuple[str, object]]:
    f, o = facts["fmt"], facts["outcomes"]
    test = (f"We tested it on purchases from {f['test_start']} to {f['test_end']}, after the training period. "
            f"For {f['svd_hit']} of {f['eligible_customers']} customers with purchase history, at least one product "
            "they bought for the first time was among their 10 personalised suggestions, against "
            f"{f['baseline_hit']} for best sellers")
    if o["svd_beats_baseline"]:
        lead = f"Pilot Truncated SVD with {f['k']} components for the \"also bought\" box. " + test
        lead += (f": an estimated gain of {f['svd_vs_baseline_diff']} points (95% interval {f['svd_vs_baseline_ci']})."
                 if "svd_vs_baseline_ci" in f else ".")
    else:
        lead = ("Keep best sellers in the box for now: the model did not beat them. "
                f"We tried Truncated SVD with {f['k']} components. " + test + ".")
    lead += (f" PCA scored {f['pca_hit']}, and this test does not establish a difference between the two."
             if o["svd_pca_difference_not_established"] else f" PCA scored {f['pca_hit']}.")
    method = (
        f"Each cell of the customer x product table is log(1 + total quantity), and products need at least "
        f"{f['min_buyers']} buyers before the test period, leaving {f['sparsity']} of cells empty. SVD compresses "
        "this table as it is. PCA instead describes differences from the average customer. We fixed "
        f"{f['k']} components before testing, from a curve on training data only. The full table needs "
        f"{f['dense_mb']} and sparse matrix storage {f['sparse_mb']}. The model's customer and product tables "
        f"need {f['svd_factor_mb']}, {f['dense_to_factor_ratio']} times less than the full table"
        + (" but more than sparse storage. " if o["factors_exceed_csr"] else " and less than sparse storage. ")
        + f"From the prepared purchase table, fitting and finding the 10 most similar products for all "
        f"{f['products']} products took {f['nightly_total']} on {f['machine']}. Raw-data cleaning is extra, and "
        "production times will differ."
    )
    loss = (
        f"Rebuilt from {f['k']} components, the log-transformed table has a relative reconstruction error of "
        f"{f['svd_error']} (Frobenius norm). It measures how roughly the model approximates purchase patterns, "
        "not a share of purchases or quantities kept. Suggestions follow broad patterns and may miss niche items."
    )
    groups = []
    for s in facts["segments"]:
        groups.append(f"<b>{esc(s['name'])}</b>: {fmt_int(s['customers'])} customers "
                      f"({s['share'] * 100:.0f}%), e.g. {esc(sentence_case(s['example_product']))}.")
    count_word = {3: "Three", 4: "Four", 5: "Five", 6: "Six"}.get(len(groups), str(len(groups)))
    group_intro = f"{count_word} groups from the first two model components, named from their products:"
    group_tail = (
        f"All groups are mainly UK-based ({f['uk_min']} to {f['uk_max']} of customers); some have a larger "
        "overseas share. Treat the groups as campaign ideas to test."
        if o["uk_dominates_every_segment"] and f["uk_min"] else "Treat the groups as campaign ideas to test."
    )
    risks = [
        f"Coverage: {f['eligible_customers']} of {f['total_test_customers']} test customers "
        f"({f['eligible_customer_coverage_pct']}) had history and a catalogue product to score. "
        f"{f['test_products_in_catalogue']} of {f['test_products']} distinct test products "
        f"({f['product_coverage_pct']}) are in the catalogue, and {f['pair_coverage_pct']} of new "
        "customer-product pairs could be scored.",
        "Product-page lists work for any visitor on a catalogue product, new customers included. Personalised "
        f"suggestions need purchase history, and new products or those with under {f['min_buyers']} buyers need "
        "best sellers as a fallback.",
        "The offline hit rate does not prove extra sales. Next: an A/B pilot against best sellers measuring "
        "clicks and orders, a marketing review of the group names, and weekly coverage tracking.",
    ]
    return [
        ("heading", "Recommendation"), ("para", esc(lead)),
        ("heading", "Method, size and speed"), ("para", esc(method)),
        ("heading", "What compression loses"), ("para", esc(loss)),
        ("heading", "Customer groups for marketing"), ("para", esc(group_intro)),
        ("bullets", groups), ("para", esc(group_tail)),
        ("heading", "Risks and next steps"), ("bullets", [esc(r) for r in risks]),
    ]


def esc(value: str) -> str:
    return value.replace("&", "&amp;").replace("<b>", "\x00").replace("</b>", "\x01") \
        .replace("<", "&lt;").replace(">", "&gt;").replace("\x00", "<b>").replace("\x01", "</b>")


def plain(markup: str) -> str:
    return re.sub(r"<[^>]+>", "", markup).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


# Embedded so every viewer shows the same sans face as the slides; base-14 fonts are not embedded.
MEMO_FONTS = [
    ("/System/Library/Fonts/HelveticaNeue.ttc", 0, "/System/Library/Fonts/HelveticaNeue.ttc", 1),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0, "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
]


def register_memo_fonts() -> tuple[str, str]:
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for regular, regular_index, bold, bold_index in MEMO_FONTS:
        if Path(regular).is_file() and Path(bold).is_file():
            pdfmetrics.registerFont(TTFont("MemoSans", regular, subfontIndex=regular_index))
            pdfmetrics.registerFont(TTFont("MemoSans-Bold", bold, subfontIndex=bold_index))
            addMapping("MemoSans", 0, 0, "MemoSans")
            addMapping("MemoSans", 1, 0, "MemoSans-Bold")
            return "MemoSans", "MemoSans-Bold"
    raise InputError("No embeddable sans font found (looked for Helvetica Neue and Arial)")


def render_memo_pdf(facts: dict, path: Path) -> dict:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import HRFlowable, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

    regular, bold = register_memo_fonts()
    navy, teal, ink, muted = (colors.HexColor(c) for c in ("#1B2A4A", "#0F7C80", "#1F2937", "#4B5563"))
    body = ParagraphStyle("body", fontName=regular, fontSize=10.5, leading=13.1, textColor=ink,
                          alignment=TA_LEFT, spaceAfter=2)
    heading = ParagraphStyle("heading", fontName=bold, fontSize=11.5, leading=14, textColor=navy,
                             spaceBefore=5, spaceAfter=1)
    title = ParagraphStyle("title", fontName=bold, fontSize=22, leading=26, textColor=navy)
    meta = ParagraphStyle("meta", parent=body, fontSize=10, leading=12.6, textColor=muted, spaceAfter=0)
    source = ParagraphStyle("source", parent=body, fontSize=8.5, leading=10.5, textColor=muted)

    story = [
        Paragraph("Customers Like You", title),
        Spacer(1, 3),
        Paragraph("<b>To:</b> Sarah Okafor, Head of E-Commerce", meta),
        Paragraph("<b>From:</b> Data &amp; ML Team: Alex Rees (Data Engineer) and Jiaxiao Chen (ML Engineer)", meta),
        Paragraph("<b>About:</b> The \"Customers who bought this also bought\" box and customer groups", meta),
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=1.2, color=teal, spaceBefore=2, spaceAfter=2),
    ]
    words = []
    for kind, content in memo_blocks(facts):
        if kind == "heading":
            story.append(Paragraph(content, heading))
            words.append(content)
        elif kind == "para":
            story.append(Paragraph(content, body))
            words.append(plain(content))
        else:
            items = [ListItem(Paragraph(item, body), leftIndent=13, bulletFontName=regular) for item in content]
            if kind == "numbers":
                lists = ListFlowable(items, bulletType="1", leftIndent=13, bulletFontName=bold,
                                     bulletFontSize=10.5, bulletColor=teal)
            else:
                lists = ListFlowable(items, bulletType="bullet", start="•", leftIndent=13,
                                     bulletFontName=regular, bulletFontSize=9, bulletColor=teal)
            story.append(lists)
            words.extend(plain(item) for item in content)
    story += [Spacer(1, 5), HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#D1D5DB")),
              Paragraph("Data: " + esc(facts["citation"]), source)]

    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.48 * inch, bottomMargin=0.42 * inch, title="Customers Like You",
                            author="DS5020 Data & ML Team", subject="Memo to Sarah Okafor")
    doc.build(story)
    all_text = " ".join(words)
    return {"word_count": len(re.findall(r"[\w'’()./+-]+", all_text)), "text": all_text}


def pdftoppm_path() -> str:
    override = os.environ.get("PDFTOPPM")
    if override:
        return override
    if BUNDLED_PDFTOPPM.is_file():
        return str(BUNDLED_PDFTOPPM)
    found = shutil.which("pdftoppm")
    if not found:
        raise InputError("pdftoppm not found; set PDFTOPPM to a Poppler pdftoppm executable")
    return found


def unique_path(path: Path, stamp: str) -> Path:
    candidate = path.with_name(f"{path.stem}-{stamp}{path.suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{stamp}-{counter}{path.suffix}")
        counter += 1
    return candidate


def build_memo(output: Path | None, replace: bool, reports: Path, draft_only: bool = False) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise InputError("pypdf is required to check the page count (pip install pypdf)") from error

    facts = build_facts(reports, require_map=not draft_only)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work = ARTIFACTS / "memo"
    work.mkdir(parents=True, exist_ok=True)
    if not draft_only:
        write_facts(facts, DEFAULT_FACTS)
    draft = work / f"memo-draft-{stamp}.pdf"
    info = render_memo_pdf(facts, draft)

    reader = PdfReader(str(draft))
    pages = len(reader.pages)
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    problems = []
    if pages != 1:
        problems.append(f"memo has {pages} pages; it must fit on one")
    if info["word_count"] > MEMO_WORD_LIMIT:
        problems.append(f"memo has {info['word_count']} words; limit is {MEMO_WORD_LIMIT}")
    privacy = privacy_findings(extracted)
    problems += privacy
    if re.search(r"\b(TODO|TBD|XXX|placeholder)\b", extracted, re.I):
        problems.append("memo contains placeholder text")

    prefix = work / f"memo-draft-{stamp}"
    subprocess.run([pdftoppm_path(), "-png", "-r", "110", str(draft), str(prefix)], check=True)
    renders = sorted(str(p.relative_to(ROOT)) for p in work.glob(f"memo-draft-{stamp}*.png"))

    report = {
        "draft": rel(draft), "pages": pages, "word_count": info["word_count"], "renders": renders,
        "privacy_scan": {"mode": private_ids()[2], "findings": privacy},
        "facts_warnings": facts["warnings"], "problems": problems,
        "inputs": facts["inputs"], "draft_sha256": sha256(draft),
    }
    if problems:
        (work / f"memo-{stamp}.validation.json").write_text(json.dumps(report, indent=2) + "\n")
        raise InputError("Memo failed validation (draft kept privately):\n  " + "\n  ".join(problems))

    if draft_only:
        report.update(final=None, note="draft only: nothing published")
        (work / f"memo-{stamp}.validation.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    target = (output or DEFAULT_MEMO).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not replace:
        target = unique_path(target, stamp)
    staged = target.with_name(f".{target.name}.{stamp}.staged")
    shutil.copyfile(draft, staged)
    if sha256(staged) != report["draft_sha256"]:
        staged.unlink()
        raise InputError("Staged memo bytes differ from the validated draft")
    staged.replace(target)
    report.update(final=rel(target), final_sha256=sha256(target), replaced_existing=replace)
    (work / f"memo-{stamp}.validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "facts", "memo", "scan"])
    parser.add_argument("--reports", type=Path, default=REPORTS, help="reports directory (default reports/)")
    parser.add_argument("--output", type=Path, help="output path for facts or memo")
    parser.add_argument("--input", type=Path, help="scan: UTF-8 text file to check for customer IDs")
    parser.add_argument("--draft-only", action="store_true",
                        help="memo: build, check and render a private draft in tmp/artifacts/memo only")
    parser.add_argument("--replace", action="store_true",
                        help="memo: replace an existing output file with the newly validated memo")
    args = parser.parse_args(argv)
    try:
        if args.command == "scan":
            if not args.input or not args.input.is_file():
                raise InputError("scan needs --input pointing to a text file")
            findings = privacy_findings(args.input.read_text(encoding="utf-8"))
            print(json.dumps({"mode": private_ids()[2], "findings": findings}))
            return 3 if findings else 0
        if args.command == "memo":
            report = build_memo(args.output, args.replace, args.reports, args.draft_only)
            print(json.dumps({k: report[k] for k in ("final", "pages", "word_count", "renders", "facts_warnings")},
                             indent=2))
        else:
            facts = build_facts(args.reports)
            if args.command == "facts":
                write_facts(facts, args.output or DEFAULT_FACTS)
                print(f"Wrote {rel(args.output or DEFAULT_FACTS)}")
            print(json.dumps({"fmt": facts["fmt"], "outcomes": facts["outcomes"],
                              "warnings": facts["warnings"]}, indent=2, ensure_ascii=False))
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
