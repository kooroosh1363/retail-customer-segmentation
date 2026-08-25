# DS-04 — Retail Customer Segmentation

Portfolio-grade unsupervised learning project that converts public transaction-level retail data into customer-level behavioral segments.

## What this project demonstrates

- reproducible acquisition from the official UCI Machine Learning Repository
- transaction cleaning with explicit cancellation, missing-customer, non-positive value and duplicate policies
- customer-level RFM feature engineering
- log1p transforms and StandardScaler preprocessing
- KMeans model comparison across multiple cluster counts
- silhouette score plus cluster-size safeguards for model selection
- repeated-subsample stability analysis with Adjusted Rand Index
- PCA as a diagnostic/visualization artifact rather than a clustering requirement
- business-oriented segment profiling with post-hoc descriptive persona labels
- pytest and GitHub Actions CI

## Data

The project uses the UCI **Online Retail** dataset: 541,909 transaction rows from a UK-based non-store retailer between December 2010 and December 2011.

See `DATA_SOURCE.md`, `DATA_DICTIONARY.md`, and `METHOD_CARD.md` for provenance, feature definitions, assumptions, and claim boundaries.

## Architecture

```text
official UCI Online Retail ZIP
    -> download + schema validation
    -> clean transaction rows
    -> derive line revenue
    -> aggregate customer RFM features
    -> log1p skewed RFM variables
    -> StandardScaler
    -> KMeans candidates (k = 2..8)
    -> choose by silhouette with minimum-cluster-size guard
    -> stability audit with repeated 80% subsamples + ARI
    -> PCA diagnostic coordinates
    -> customer assignments + cluster profiles + personas
    -> JSON/CSV artifacts
    -> pytest + GitHub Actions CI
```

## RFM features

- `recency_days`: days since the customer's latest valid purchase relative to the day after the dataset's latest purchase
- `frequency_orders`: number of unique valid invoices
- `monetary_value`: total valid purchase revenue
- `items_purchased`: total units purchased
- `avg_order_value`: monetary value divided by unique order count

The clustering core uses `recency_days`, `frequency_orders`, and `monetary_value`. Extra customer metrics are retained for business profiling rather than used to force additional clustering dimensions.

## Model-selection policy

KMeans candidates from `k=2` through `k=8` are fitted with deterministic seeds. The primary quality metric is silhouette score. A candidate must also keep every cluster above a minimum share of the customer base so a tiny outlier group cannot win purely on geometric separation.

The selected solution is then audited for stability across ten repeated 80% customer subsamples. Each subsample is reclustered and compared with the full-data labels on those same customers using Adjusted Rand Index, which is invariant to arbitrary cluster-label numbering.

## Persona policy

Persona names are added only after clustering. Cluster-level median recency, frequency, and monetary behavior is ranked from stronger to weaker, then descriptive names are spread across a high-to-low engagement ladder. These names are interpretation aids rather than learned classes or ground-truth customer types, and they never affect model fitting or cluster selection.

## Claim boundary

These are descriptive behavioral segments derived from one historical retail dataset. Cluster labels are not natural customer types, causal categories, or evidence that the same segment structure will persist in another market or time period.

## Generated artifacts

Running the pipeline writes ignored outputs to `artifacts/`:

- `metrics.json`
- `customer_segments.csv`
- `cluster_profiles.csv`
- `cluster_selection.csv`
- `stability_runs.csv`
- `pca_coordinates.csv`
- `model.joblib`

## Run locally

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.segment
```
