from src.data import build_customer_features, clean_transactions, load_raw


def test_raw_and_clean_contract():
    raw = load_raw()
    assert len(raw) == 541_909
    assert list(raw.columns) == [
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country",
    ]

    clean, audit = clean_transactions(raw)
    assert 380_000 < len(clean) < 410_000
    assert audit["missing_customer_rows_removed"] > 100_000
    assert audit["cancelled_rows_removed"] > 5_000
    assert audit["exact_duplicates_removed"] > 0
    assert clean["CustomerID"].notna().all()
    assert (clean["Quantity"] > 0).all()
    assert (clean["UnitPrice"] > 0).all()
    assert (clean["line_revenue"] > 0).all()


def test_customer_features_are_one_row_per_customer():
    clean, _ = clean_transactions(load_raw())
    customers, snapshot_date = build_customer_features(clean)
    assert 4_000 < len(customers) < 4_500
    assert customers["CustomerID"].is_unique
    assert snapshot_date > clean["InvoiceDate"].max()
    assert (customers["recency_days"] >= 1).all()
    assert (customers["frequency_orders"] >= 1).all()
    assert (customers["monetary_value"] > 0).all()
    assert (customers["avg_order_value"] > 0).all()
