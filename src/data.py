from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import io

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_XLSX = RAW_DIR / "Online Retail.xlsx"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
EXPECTED_RAW_ROWS = 541_909
EXPECTED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]


def download_raw() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_XLSX.exists():
        return RAW_XLSX

    response = requests.get(UCI_ZIP_URL, timeout=120)
    response.raise_for_status()
    with ZipFile(io.BytesIO(response.content)) as zf:
        matches = [name for name in zf.namelist() if name.lower().endswith(".xlsx")]
        if len(matches) != 1:
            raise ValueError(f"Expected one XLSX in UCI archive, found: {matches}")
        RAW_XLSX.write_bytes(zf.read(matches[0]))
    return RAW_XLSX


def load_raw() -> pd.DataFrame:
    path = download_raw()
    df = pd.read_excel(path, engine="openpyxl")
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected schema: {list(df.columns)}")
    if len(df) != EXPECTED_RAW_ROWS:
        raise ValueError(f"Unexpected raw row count: {len(df)}")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="raise")
    return df


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    audit = {"raw_rows": int(len(work))}

    customer_mask = work["CustomerID"].notna()
    audit["missing_customer_rows_removed"] = int((~customer_mask).sum())
    work = work.loc[customer_mask].copy()

    work["InvoiceNo"] = work["InvoiceNo"].astype(str)
    cancel_mask = work["InvoiceNo"].str.startswith("C", na=False)
    audit["cancelled_rows_removed"] = int(cancel_mask.sum())
    work = work.loc[~cancel_mask].copy()

    positive_mask = (work["Quantity"] > 0) & (work["UnitPrice"] > 0)
    audit["non_positive_rows_removed"] = int((~positive_mask).sum())
    work = work.loc[positive_mask].copy()

    before_dedup = len(work)
    work = work.drop_duplicates().copy()
    audit["exact_duplicates_removed"] = int(before_dedup - len(work))

    work["CustomerID"] = work["CustomerID"].astype("int64").astype(str)
    work["line_revenue"] = work["Quantity"].astype(float) * work["UnitPrice"].astype(float)
    if not np.isfinite(work["line_revenue"]).all() or (work["line_revenue"] <= 0).any():
        raise ValueError("Cleaned line revenue must be finite and positive")

    audit["clean_rows"] = int(len(work))
    audit["customers"] = int(work["CustomerID"].nunique())
    audit["invoices"] = int(work["InvoiceNo"].nunique())
    audit["latest_purchase"] = work["InvoiceDate"].max().isoformat()
    return work.reset_index(drop=True), audit


def build_customer_features(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    snapshot_date = clean["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    agg = clean.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        frequency_orders=("InvoiceNo", "nunique"),
        monetary_value=("line_revenue", "sum"),
        items_purchased=("Quantity", "sum"),
        country=("Country", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
    )
    agg["recency_days"] = (snapshot_date - agg["last_purchase"].dt.normalize()).dt.days.astype(int)
    agg["avg_order_value"] = agg["monetary_value"] / agg["frequency_orders"]
    agg = agg.reset_index()

    if (agg[["recency_days", "frequency_orders", "monetary_value"]].min() < 0).any():
        raise ValueError("RFM values must be non-negative")
    if agg["CustomerID"].duplicated().any():
        raise ValueError("Customer feature table must contain one row per customer")
    return agg, snapshot_date
