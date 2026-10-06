Abstract Optimization Model

Index Sets:
- \( I \): Set of products, indexed by \( i \).

Parameters:
- \( r_i \): Per-unit revenue for product \( i \). (Source: MobileSalesDataset.csv, column "Revenue")
- \( d_i \): Expected demand for product \( i \) during the sales cycle. (Source: MobileSalesDataset.csv, column "Demand")
- \( s_i \): Initial inventory available for product \( i \). (Source: MobileSalesDataset.csv, column "Initial Inventory")

Decision Variables:
- \( x_i \): Number of units of product \( i \) fulfilled (orders fulfilled), integer, \( x_i \geq 0 \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled orders.)

Constraints:
1. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
   (Cannot fulfill more than available inventory.)

2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
   (Cannot fulfill more than realized demand.)

3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]
   (Fulfillment quantities must be non-negative integers.)

Data Mapping:
- Index set \( I \): All rows in MobileSalesDataset.csv (table_id: file_0_view_0)
- \( r_i \): "Revenue" column, MobileSalesDataset.csv (table_id: file_0_view_0)
- \( d_i \): "Demand" column, MobileSalesDataset.csv (table_id: file_0_view_0)
- \( s_i \): "Initial Inventory" column, MobileSalesDataset.csv (table_id: file_0_view_0)

No data values or record counts are included; all definitions are symbolic and mapped to their exact sources.