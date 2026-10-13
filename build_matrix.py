"""Rebuild the customer x product training matrix from the raw Online Retail II file.

    python build_matrix.py

Runs DE1 to DE5 in order (profile, clean, time split, sparse matrix, matrix report).
The steps and their order are the ones agreed in notebooks/data_exploration.ipynb.

Private outputs (customer IDs) go to the git-ignored data/processed/ folder.
Aggregate reports with no customer IDs go to reports/.
"""

import os

# Pin numerical libraries to one thread so timings are repeatable (set before numpy loads).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "online_retail_II.xlsx"
RAW_ZIP = ROOT / "online+retail+ii.zip"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

TEST_MONTHS = 2
MIN_BUYERS = 10
PURCHASE_VALUE = "log"

# Canonical column names, keyed by the lower-case name with spaces and underscores removed.
COLUMN_ALIASES = {
    "invoice": "Invoice", "invoiceno": "Invoice",
    "stockcode": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoicedate": "InvoiceDate",
    "price": "Price", "unitprice": "Price",
    "customerid": "Customer ID",
    "country": "Country",
}
REQUIRED_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity",
                    "InvoiceDate", "Price", "Customer ID", "Country"]

# Postage, fees, discounts, samples, adjustments, tests and vouchers (DE2 rule 4).
# A letter on the end of a numeric code is a product variant, so 85123A stays a product.
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "m", "C2", "D", "S", "BANK CHARGES", "AMAZONFEE", "B",
    "ADJUST", "ADJUST2", "CRUK", "TEST001", "TEST002", "PADS",
}
GIFT_VOUCHER_PREFIX = "gift_0001_"


# ---------------------------------------------------------------- DE1 load and profile

def ensure_raw_file(raw_path=RAW_FILE, zip_path=RAW_ZIP):
    """Return the raw workbook path, extracting only online_retail_II.xlsx from the zip if needed."""
    raw_path = Path(raw_path)
    if raw_path.exists():
        return raw_path
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path} and {zip_path}. Download the zip from "
            "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
        )
    with zipfile.ZipFile(zip_path) as archive:
        if raw_path.name not in archive.namelist():
            raise FileNotFoundError(f"{zip_path} does not contain {raw_path.name}")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        partial = raw_path.with_suffix(".partial")
        # Read the one expected member by name and write it to a fixed path, so no
        # path inside the archive can choose where files are written.
        with archive.open(raw_path.name) as src, open(partial, "wb") as dst:
            shutil.copyfileobj(src, dst)
        partial.replace(raw_path)
    print(f"Extracted {raw_path.name} from {zip_path.name}")
    return raw_path


def canonicalize_columns(df):
    """Rename known variants (InvoiceNo, UnitPrice, CustomerID, ...) to the canonical names."""
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "").replace("_", "")
        renamed[col] = COLUMN_ALIASES.get(key, str(col).strip())
    df = df.rename(columns=renamed)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing columns: {missing}")
    return df


def load_raw(path):
    """Load and stack every sheet. Returns the combined frame and the rows per sheet."""
    sheets = pd.read_excel(path, sheet_name=None)
    frames = [canonicalize_columns(sheet) for sheet in sheets.values()]
    df = pd.concat(frames, ignore_index=True)
    sheet_counts = pd.DataFrame({"sheet": list(sheets), "rows": [len(s) for s in sheets.values()]})
    sheet_counts.loc[len(sheet_counts)] = ["Total", int(sheet_counts["rows"].sum())]
    return df, sheet_counts


def profile_columns(df):
    """DE1 profile: one row per column with type, present, missing and distinct counts."""
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).to_numpy(),
        "non_null": df.notna().sum().to_numpy(),
        "missing": df.isna().sum().to_numpy(),
        "missing_pct": (df.isna().mean().to_numpy() * 100).round(2),
        "n_unique": [df[col].nunique(dropna=True) for col in df.columns],
    })


# ---------------------------------------------------------------- DE2 clean

def is_non_product(codes):
    return codes.isin(NON_PRODUCT_CODES) | codes.str.startswith(GIFT_VOUCHER_PREFIX)


CLEANING_RULES = [
    ("Cancellations", "Invoice starts with C",
     lambda d: d["Invoice"].str.startswith("C")),
    ("Non-positive quantity", "Quantity <= 0",
     lambda d: d["Quantity"] <= 0),
    ("Non-positive price", "Price <= 0",
     lambda d: d["Price"] <= 0),
    ("Non-product codes", "Postage, fees, discounts, samples, adjustments, tests, vouchers",
     lambda d: is_non_product(d["StockCode"])),
    ("Missing customer ID", "Customer ID is null",
     lambda d: d["Customer ID"].isna()),
]


def clean_transactions(df):
    """Apply the DE2 rules in order. Each removed row is counted under the first rule it hits."""
    clean = df.copy()
    clean["Invoice"] = clean["Invoice"].astype(str).str.strip()
    clean["StockCode"] = clean["StockCode"].astype(str).str.strip()

    log_rows = []
    for rule, reason, mask_fn in CLEANING_RULES:
        drop = mask_fn(clean)
        clean = clean.loc[~drop].copy()
        log_rows.append({"rule": rule, "reason": reason,
                         "rows_removed": int(drop.sum()), "rows_remaining": len(clean)})
    log = pd.DataFrame(log_rows)
    log.loc[len(log)] = {"rule": "Total removed", "reason": "",
                         "rows_removed": int(log["rows_removed"].sum()), "rows_remaining": len(clean)}
    clean["Customer ID"] = clean["Customer ID"].astype("int64")
    return clean, log


# ---------------------------------------------------------------- DE3 split

def split_by_time(clean, months=TEST_MONTHS):
    """Train is everything before the cutoff. Test is only the test-period purchases of
    products the customer had never bought before the cutoff (R2)."""
    cutoff = clean["InvoiceDate"].max() - pd.DateOffset(months=months)
    train = clean.loc[clean["InvoiceDate"] < cutoff].copy()
    future = clean.loc[clean["InvoiceDate"] >= cutoff].copy()

    seen = pd.MultiIndex.from_frame(train[["Customer ID", "StockCode"]].drop_duplicates())
    future_pairs = pd.MultiIndex.from_frame(future[["Customer ID", "StockCode"]])
    bought_before = future_pairs.isin(seen)
    test = future.loc[~bought_before].copy()
    repeats = future.loc[bought_before].copy()
    return train, test, repeats, cutoff


def split_log(train, test, repeats):
    rows = []
    for name, frame in [("Train (before cutoff)", train), ("Test (new products)", test),
                        ("Test-period repeats excluded", repeats)]:
        rows.append({
            "split": name, "rows": len(frame),
            "customers": frame["Customer ID"].nunique(),
            "products": frame["StockCode"].nunique(),
            "start": frame["InvoiceDate"].min(), "end": frame["InvoiceDate"].max(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- DE4 matrix

def purchase_value(quantity, weighting=PURCHASE_VALUE):
    """Turn summed quantities into the stored purchase value: log(1 + q), raw count or 1/0."""
    quantity = np.asarray(quantity, dtype=np.float64)
    if weighting == "log":
        return np.log1p(quantity)
    if weighting == "count":
        return quantity
    if weighting == "binary":
        return np.ones_like(quantity)
    raise ValueError(f"Unknown weighting {weighting!r}")


def aggregate_quantities(train):
    """Total quantity for each customer and product (aggregate first, transform later)."""
    return (train.groupby(["Customer ID", "StockCode"], as_index=False)["Quantity"].sum()
            .rename(columns={"Quantity": "quantity"}))


def filter_products(quantity, min_buyers=MIN_BUYERS):
    """Keep products bought by at least min_buyers distinct training customers."""
    buyers = quantity.groupby("StockCode")["Customer ID"].nunique()
    keep = buyers.index[buyers >= min_buyers]
    return quantity.loc[quantity["StockCode"].isin(keep)].reset_index(drop=True)


def build_matrices(quantity, weighting=PURCHASE_VALUE):
    """Sparse CSR matrices with sorted, stable row (customer) and column (product) order.

    Customers whose products were all filtered out do not appear, so there are no empty rows.
    Returns the weighted matrix, the raw-quantity matrix and the two index arrays.
    """
    customer_ids = np.array(sorted(quantity["Customer ID"].unique()))
    product_ids = np.array(sorted(quantity["StockCode"].unique()), dtype=object)
    rows = np.searchsorted(customer_ids, quantity["Customer ID"].to_numpy())
    cols = pd.Index(product_ids).get_indexer(quantity["StockCode"])
    shape = (len(customer_ids), len(product_ids))
    qty = quantity["quantity"].to_numpy(dtype=np.float64)
    Q = sp.csr_matrix((qty, (rows, cols)), shape=shape)
    A = sp.csr_matrix((purchase_value(qty, weighting), (rows, cols)), shape=shape)
    Q.sort_indices()
    A.sort_indices()
    return A, Q, customer_ids, product_ids


def product_descriptions(train, product_ids):
    """Most frequent training description per product; ties go to the alphabetically first
    name, and a product with no description falls back to its stock code."""
    desc = train.loc[train["StockCode"].isin(product_ids), ["StockCode", "Description"]].dropna()
    desc = desc.assign(Description=desc["Description"].astype(str).str.strip())
    desc = desc.loc[desc["Description"] != ""]
    counts = desc.groupby(["StockCode", "Description"]).size().reset_index(name="n")
    counts = counts.sort_values(["StockCode", "n", "Description"], ascending=[True, False, True])
    best = counts.drop_duplicates("StockCode").set_index("StockCode")["Description"]
    return pd.Series([best.get(code, code) for code in product_ids], index=product_ids)


def dominant_countries(train, customer_ids):
    """Each customer's most frequent training country (ties: alphabetical)."""
    counts = train.groupby(["Customer ID", "Country"]).size().reset_index(name="n")
    counts = counts.sort_values(["Customer ID", "n", "Country"], ascending=[True, False, True])
    best = counts.drop_duplicates("Customer ID").set_index("Customer ID")["Country"]
    return best.reindex(customer_ids).to_numpy()


# ---------------------------------------------------------------- DE5 report

def matrix_memory(A):
    n_cells = A.shape[0] * A.shape[1]
    return {
        "dense_bytes": int(n_cells * A.dtype.itemsize),
        "sparse_bytes": int(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes),
        "sparse_data_bytes": int(A.data.nbytes),
        "sparse_indices_bytes": int(A.indices.nbytes),
        "sparse_indptr_bytes": int(A.indptr.nbytes),
        "sparsity_pct": 100 * (1 - A.nnz / n_cells),
    }


def matrix_report(A, quantity_all, quantity_kept, train, test, repeats, clean, raw_rows, cutoff):
    memory = matrix_memory(A)
    test_customers = set(test["Customer ID"])
    matrix_products = set(quantity_kept["StockCode"])
    return {
        "shape": list(A.shape),
        "customers": A.shape[0],
        "products": A.shape[1],
        "nonzeros": int(A.nnz),
        **memory,
        "dense_mb": memory["dense_bytes"] / 1e6,
        "sparse_mb": memory["sparse_bytes"] / 1e6,
        "memory_units": "decimal MB (1 MB = 1,000,000 bytes)",
        "dtype": str(A.dtype),
        "index_dtype": str(A.indices.dtype),
        "purchase_value": "log(1 + total quantity per customer and product)",
        "min_buyers_per_product": MIN_BUYERS,
        "raw_rows": int(raw_rows),
        "clean_rows": len(clean),
        "train_rows": len(train),
        "test_rows": len(test),
        "excluded_repeat_rows": len(repeats),
        "cutoff": str(cutoff),
        "test_months": TEST_MONTHS,
        "train_start": str(train["InvoiceDate"].min()),
        "train_end": str(train["InvoiceDate"].max()),
        "test_start": str(test["InvoiceDate"].min()),
        "test_end": str(test["InvoiceDate"].max()),
        "train_customers": int(train["Customer ID"].nunique()),
        "train_products": int(train["StockCode"].nunique()),
        "train_customer_product_pairs": len(quantity_all),
        "products_dropped_by_min_buyers": int(quantity_all["StockCode"].nunique() - A.shape[1]),
        "pairs_dropped_by_min_buyers": int(len(quantity_all) - len(quantity_kept)),
        "customers_dropped_with_no_kept_products": int(train["Customer ID"].nunique() - A.shape[0]),
        "test_customers": len(test_customers),
        "test_customers_without_training_history": len(test_customers - set(train["Customer ID"])),
        "test_rows_product_not_in_training": int((~test["StockCode"].isin(set(train["StockCode"]))).sum()),
        "test_rows_product_not_in_matrix": int((~test["StockCode"].isin(matrix_products)).sum()),
    }


def _save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW_FILE)
    parser.add_argument("--zip", type=Path, default=RAW_ZIP)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)
    out, reports = args.processed_dir, args.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    print("DE1 load and profile (reading Excel takes a minute or two)")
    raw, sheet_counts = load_raw(ensure_raw_file(args.raw, args.zip))
    print(f"  {len(raw):,} raw rows")
    _save_csv(sheet_counts, reports / "sheet_counts.csv")
    _save_csv(profile_columns(raw), reports / "data_profile.csv")

    print("DE2 clean")
    clean, cleaning_log = clean_transactions(raw)
    print(f"  {len(clean):,} clean rows")
    _save_csv(cleaning_log, reports / "cleaning_log.csv")

    print("DE3 time split")
    train, test, repeats, cutoff = split_by_time(clean)
    print(f"  cutoff {cutoff}: {len(train):,} train rows, {len(test):,} novel test rows, "
          f"{len(repeats):,} repeat rows excluded")
    assert train["InvoiceDate"].max() < cutoff <= test["InvoiceDate"].min()
    assert train.merge(test, on=["Customer ID", "StockCode"]).empty
    _save_csv(train, out / "train.csv")
    _save_csv(test, out / "test.csv")
    _save_csv(split_log(train, test, repeats), reports / "split_log.csv")

    print("DE4 sparse matrix from training rows only")
    quantity_all = aggregate_quantities(train)
    quantity = filter_products(quantity_all)
    A, Q, customer_ids, product_ids = build_matrices(quantity)
    assert A.nnz == len(quantity) and (np.diff(A.indptr) > 0).all()
    sp.save_npz(out / "customer_product.npz", A)
    sp.save_npz(out / "customer_product_quantity.npz", Q)
    _save_csv(pd.DataFrame({"row": range(len(customer_ids)), "customer_id": customer_ids}),
              out / "customer_index.csv")
    _save_csv(pd.DataFrame({"row": range(len(customer_ids)), "customer_id": customer_ids,
                            "country": dominant_countries(train, customer_ids)}),
              out / "customer_countries.csv")
    _save_csv(pd.DataFrame({"col": range(len(product_ids)), "stock_code": product_ids,
                            "description": product_descriptions(train, product_ids).to_numpy()}),
              out / "product_index.csv")

    print("DE5 matrix report")
    report = matrix_report(A, quantity_all, quantity, train, test, repeats, clean, len(raw), cutoff)
    (reports / "matrix_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"  {A.shape[0]:,} x {A.shape[1]:,}, {A.nnz:,} non-zeros, "
          f"{report['sparsity_pct']:.2f}% sparse, dense {report['dense_mb']:.1f} MB vs "
          f"sparse {report['sparse_mb']:.1f} MB")
    print(f"  wrote {(reports / 'matrix_report.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
