# Data Source

## Canonical source

This project uses the **Online Retail** dataset from the UCI Machine Learning Repository.

Canonical page:
https://archive.ics.uci.edu/dataset/352/online+retail

Direct official archive used by the pipeline:
https://archive.ics.uci.edu/static/public/352/online+retail.zip

UCI describes this as transaction data for a UK-based registered non-store online retailer, covering transactions from 01 December 2010 through 09 December 2011. The dataset contains 541,909 rows.

## Raw columns

- `InvoiceNo`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `CustomerID`
- `Country`

## Cleaning policy

The segmentation model is built only from completed, attributable, positive-value purchase activity. The pipeline therefore:

1. removes rows with missing `CustomerID` because they cannot be aggregated to a known customer;
2. removes cancelled invoices identified by invoice numbers beginning with `C`;
3. removes rows with non-positive `Quantity` or `UnitPrice`;
4. removes exact duplicate transaction rows;
5. computes `line_revenue = Quantity * UnitPrice`;
6. aggregates valid transactions to customer-level features.

Every removal count is written to `artifacts/metrics.json` so the transformation from raw transactions to customer-level clustering data remains auditable.

## Claim boundaries

- This is historical retail data, not a current customer base.
- The company identity is not provided by UCI and should not be inferred.
- The project treats completed positive-value invoice rows as purchase behavior; it does not model returns/cancellations as a separate behavioral signal.
- Missing customer identifiers prevent those transactions from participating in customer-level segmentation.
- Segment structure is sample-specific and is not claimed to generalize automatically to another retailer or time period.
