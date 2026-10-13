"""DE checks on tiny hand-made invoice logs whose correct answers are worked out by hand."""

import zipfile

import numpy as np
import pandas as pd
import pytest

import build_matrix as bm


def lines(rows):
    """Rows of (invoice, stock code, quantity, date, price, customer, country, description)."""
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price",
                                     "Customer ID", "Country", "Description"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def test_column_aliases_are_canonicalised():
    df = pd.DataFrame(columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
                               "UnitPrice", "CustomerID", "Country"])
    assert list(bm.canonicalize_columns(df).columns) == bm.REQUIRED_COLUMNS
    with pytest.raises(ValueError):
        bm.canonicalize_columns(df.drop(columns="Country"))


def test_cleaning_rules_apply_in_order_and_keep_variants():
    raw = lines([
        ("C1001", "85123A", -2, "2011-01-01", 2.5, 1.0, "UK", "HEART"),     # cancellation (and qty<0)
        ("1002", "85123A", 0, "2011-01-01", 2.5, 1.0, "UK", "HEART"),       # quantity 0
        ("1003", "85123A", 1, "2011-01-01", 0.0, 1.0, "UK", "HEART"),       # price 0
        ("1004", "POST", 1, "2011-01-01", 18.0, 1.0, "UK", "POSTAGE"),       # non-product
        ("1005", "gift_0001_20", 1, "2011-01-01", 20.0, 1.0, "UK", "VOUCHER"),
        ("1006", "85123A", 3, "2011-01-01", 2.5, np.nan, "UK", "HEART"),    # no customer
        ("1007", "85123A", 3, "2011-01-01", 2.5, 1.0, "UK", "HEART"),       # kept
        ("1007", "85123B", 3, "2011-01-01", 2.5, 1.0, "UK", "HEART B"),     # variant kept
        ("1007", 85123, 3, "2011-01-01", 2.5, 1.0, "UK", "HEART PLAIN"),    # numeric code kept as text
        ("1008", "SP1002", 1, "2011-01-01", 2.5, 2.0, "UK", "KIDS KIT"),    # letter-first product kept
    ])
    clean, log = bm.clean_transactions(raw)
    assert log["rows_removed"].tolist() == [1, 1, 1, 2, 1, 6]
    assert log["rows_remaining"].iloc[-1] == 4
    assert sorted(clean["StockCode"]) == ["85123", "85123A", "85123B", "SP1002"]
    assert clean["Customer ID"].dtype == np.int64


def test_split_is_chronological_and_test_has_only_new_pairs():
    clean = lines([
        ("1", "A", 1, "2011-01-01 10:00", 1.0, 1, "UK", "a"),
        ("2", "B", 1, "2011-08-01 10:00", 1.0, 2, "UK", "b"),
        ("3", "A", 2, "2011-10-20 09:00", 1.0, 1, "UK", "a"),   # repeat of a training pair -> excluded
        ("4", "B", 2, "2011-10-20 09:00", 1.0, 1, "UK", "b"),   # new product for customer 1 -> test
        ("5", "C", 2, "2011-11-01 09:00", 1.0, 3, "UK", "c"),   # cold-start customer -> test
        ("6", "B", 1, "2011-12-15 10:00", 1.0, 2, "UK", "b"),   # last timestamp, repeat -> excluded
    ])
    clean["Customer ID"] = clean["Customer ID"].astype("int64")
    train, test, repeats, cutoff = bm.split_by_time(clean)
    assert cutoff == pd.Timestamp("2011-10-15 10:00")
    assert sorted(train["Invoice"]) == ["1", "2"]
    assert sorted(test["Invoice"]) == ["4", "5"]
    assert sorted(repeats["Invoice"]) == ["3", "6"]
    assert train["InvoiceDate"].max() < cutoff <= test["InvoiceDate"].min()
    assert train.merge(test, on=["Customer ID", "StockCode"]).empty


def test_row_exactly_at_cutoff_goes_to_test_period():
    clean = lines([
        ("1", "A", 1, "2011-01-01", 1.0, 1, "UK", "a"),
        ("2", "B", 1, "2011-03-01", 1.0, 1, "UK", "b"),
        ("3", "C", 1, "2011-05-01", 1.0, 1, "UK", "c"),
    ])
    train, test, _, cutoff = bm.split_by_time(clean)
    assert cutoff == pd.Timestamp("2011-03-01")
    assert test["Invoice"].tolist() == ["2", "3"]


def test_quantities_are_summed_before_log_transform():
    train = lines([
        ("1", "A", 3, "2011-01-01", 1.0, 7, "UK", "a"),
        ("2", "A", 5, "2011-02-01", 1.0, 7, "UK", "a"),
    ])
    q = bm.aggregate_quantities(train)
    A, Q, _, _ = bm.build_matrices(q)
    assert Q[0, 0] == 8
    assert A[0, 0] == pytest.approx(np.log(9))           # log(1 + 8), not log(4) + log(6)
    assert bm.purchase_value([8], "binary")[0] == 1.0
    assert bm.purchase_value([8], "count")[0] == 8.0


def test_product_filter_counts_training_buyers_and_drops_empty_customers():
    rows = [(str(i), "POPULAR", 1, "2011-01-01", 1.0, 100 + i, "UK", "pop") for i in range(10)]
    rows += [(str(20 + i), "NICHE", 1, "2011-01-01", 1.0, 100 + i, "UK", "niche") for i in range(9)]
    rows += [("99", "NICHE", 4, "2011-01-01", 1.0, 999, "UK", "niche")]   # 10th distinct NICHE buyer
    rows += [("98", "RARE", 4, "2011-01-01", 1.0, 555, "UK", "rare")]
    q = bm.aggregate_quantities(lines(rows))
    kept = bm.filter_products(q, min_buyers=10)
    assert set(kept["StockCode"]) == {"POPULAR", "NICHE"}
    A, _, customers, products = bm.build_matrices(kept)
    assert 555 not in customers                          # its only product was dropped
    assert list(products) == sorted(products) and list(customers) == sorted(customers)
    assert (np.diff(A.indptr) > 0).all()


def test_catalogue_and_indices_come_from_training_rows_only():
    clean = lines(
        [(str(i), "OLD", 1, "2011-01-01", 1.0, i, "UK", "old") for i in range(10)]
        + [(str(50 + i), "NEWTEST", 1, "2011-12-01", 1.0, i, "UK", "new") for i in range(12)]
    )
    train, test, _, _ = bm.split_by_time(clean)
    _, _, customers, products = bm.build_matrices(bm.filter_products(bm.aggregate_quantities(train)))
    assert list(products) == ["OLD"]                     # NEWTEST has 12 buyers, all in the test period
    assert len(test) == 12


def test_descriptions_use_training_mode_with_ties_and_fallback():
    train = lines([
        ("1", "A", 1, "2011-01-01", 1.0, 1, "UK", "ZEBRA MUG"),
        ("2", "A", 1, "2011-01-01", 1.0, 2, "UK", "APPLE MUG"),
        ("3", "B", 1, "2011-01-01", 1.0, 1, "UK", "BOWL"),
        ("4", "B", 1, "2011-01-01", 1.0, 2, "UK", "BOWL "),
        ("5", "B", 1, "2011-01-01", 1.0, 3, "UK", "BOWL LARGE"),
        ("6", "C", 1, "2011-01-01", 1.0, 3, "UK", None),
    ])
    desc = bm.product_descriptions(train, np.array(["A", "B", "C"], dtype=object))
    assert desc.tolist() == ["APPLE MUG", "BOWL", "C"]


def test_dominant_country_ties_are_alphabetical():
    train = lines([
        ("1", "A", 1, "2011-01-01", 1.0, 1, "France", "a"),
        ("2", "A", 1, "2011-01-01", 1.0, 1, "Belgium", "a"),
        ("3", "A", 1, "2011-01-01", 1.0, 2, "EIRE", "a"),
        ("4", "B", 1, "2011-01-01", 1.0, 2, "EIRE", "b"),
        ("5", "B", 1, "2011-01-01", 1.0, 2, "Spain", "b"),
    ])
    assert bm.dominant_countries(train, np.array([1, 2])).tolist() == ["Belgium", "EIRE"]


def test_sparse_memory_counts_values_indices_and_pointers():
    import scipy.sparse as sp
    A = sp.csr_matrix(np.array([[1.0, 0, 2], [0, 0, 3]]))
    mem = bm.matrix_memory(A)
    assert mem["dense_bytes"] == 6 * 8
    assert mem["sparse_bytes"] == 3 * 8 + 3 * A.indices.itemsize + 3 * A.indptr.itemsize
    assert mem["sparsity_pct"] == pytest.approx(50.0)


def test_zip_extraction_writes_only_the_expected_file(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("online_retail_II.xlsx", b"workbook-bytes")
        z.writestr("../evil.txt", b"should never be written")
    target = tmp_path / "raw" / "online_retail_II.xlsx"
    assert bm.ensure_raw_file(target, archive) == target
    assert target.read_bytes() == b"workbook-bytes"
    assert sorted(p.name for p in tmp_path.rglob("*")) == ["data.zip", "online_retail_II.xlsx", "raw"]
