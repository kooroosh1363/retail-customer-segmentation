# Data Dictionary

## Raw transaction fields

| Field | Meaning |
|---|---|
| InvoiceNo | Invoice identifier; cancellations are prefixed with `C` |
| StockCode | Product/item code |
| Description | Product description |
| Quantity | Units on the invoice line |
| InvoiceDate | Transaction timestamp |
| UnitPrice | Unit price in sterling |
| CustomerID | Customer identifier; missing for some transactions |
| Country | Customer/invoice country field |

## Derived transaction field

| Field | Definition |
|---|---|
| line_revenue | `Quantity * UnitPrice` after cleaning |

## Customer-level features

| Field | Definition | Clustering use |
|---|---|---|
| recency_days | Days since latest valid purchase relative to one day after the dataset's final purchase date | Yes |
| frequency_orders | Number of unique valid invoices | Yes |
| monetary_value | Sum of valid positive purchase revenue | Yes |
| items_purchased | Sum of valid positive quantities | Profiling only |
| avg_order_value | `monetary_value / frequency_orders` | Profiling only |
| country | Modal country for the customer | Profiling only |
| cluster | Numeric KMeans cluster label; label numbers are arbitrary | Output |
| persona | Human-readable business label assigned after clustering from relative RFM profile | Output |
