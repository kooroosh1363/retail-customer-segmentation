# Method Card

## Intended use

Educational/portfolio demonstration of customer segmentation using historical retail transactions aggregated to RFM behavior.

## Unit of analysis

The clustering unit is the customer, not the invoice line. Transaction rows are cleaned first and then aggregated to one row per known customer.

## Core features

The clustering model uses only:

- `recency_days`
- `frequency_orders`
- `monetary_value`

These variables are transformed with `log1p` and standardized before KMeans. The log transform reduces the influence of highly skewed frequency and monetary distributions while preserving ordering.

## Candidate models

KMeans is fitted for `k=2..8` with fixed random state and multiple initializations. The primary selection metric is silhouette score.

A minimum cluster-share guard of 5% is applied. A candidate with any cluster smaller than 5% of customers is ineligible even if it has the best silhouette score. This prevents a tiny outlier group from winning solely because it is geometrically isolated.

## Stability audit

The final `k` is evaluated over ten repeated 80% customer subsamples. Each subsample is reclustered and compared with the full-data labels on the same customers using Adjusted Rand Index (ARI). ARI is invariant to arbitrary numeric cluster-label permutations.

This audit measures robustness to sampling variation; it does not prove temporal stability.

## PCA

PCA with two components is produced only for diagnostic visualization. The KMeans model is fitted in the full standardized three-dimensional RFM space, not on the two PCA coordinates.

## Personas

Persona labels are assigned after clustering by ranking cluster-level median RFM behavior. These names are explanatory conveniences, not learned classes or universal marketing taxonomies.

## Limitations

- one historical retailer and time window;
- no external or temporal validation;
- returns/cancellations are removed rather than modeled as behavior;
- unidentified customers are excluded;
- RFM intentionally ignores product preferences, geography, margin, channel and acquisition data;
- KMeans favors roughly compact clusters in Euclidean feature space;
- silhouette and ARI are internal diagnostics, not measures of campaign ROI;
- segment names are descriptive, not causal.

## Production extension

A production segmentation system would need scheduled feature snapshots, temporal stability monitoring, segment migration analysis, business outcome validation, versioned persona definitions, drift alerts and controlled downstream campaign experiments.
