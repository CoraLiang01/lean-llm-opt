ABSTRACT Optimization Model

Index Sets:
- \( I \): Set of products classified under ‘ELE-S’, indexed by \( i \).
  (Data source: SalesStoreoverview.csv, table_id: file_0_view_0, column: Product_Reference)

Parameters:
- \( r_i \): Revenue per unit of product \( i \).
  (Data source: SalesStoreoverview.csv, table_id: file_0_view_0, column: Revenue)
- \( s_i \): Initial inventory of product \( i \).
  (Data source: SalesStoreoverview.csv, table_id: file_0_view_0, column: Initial Inventory)
- \( d_i \): Demand for product \( i \).
  (Data source: SalesStoreoverview.csv, table_id: file_0_view_0, column: Demand)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill.
  Domain: Integer, \( 0 \leq x_i \leq \min(s_i, d_i) \), \( \forall i \in I \)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units.)

Constraints:
1. Inventory and demand fulfillment:
  \[
  0 \leq x_i \leq \min(s_i, d_i) \quad \forall i \in I
  \]
  (Do not fulfill more than available inventory or demand.)

2. Integrality:
  \[
  x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
  \]

Data Mapping:
- Index set \( I \): All rows in SalesStoreoverview.csv (table_id: file_0_view_0) where Product_Reference starts with 'ELE-S'.
- Parameter \( r_i \): Revenue column, table_id: file_0_view_0.
- Parameter \( s_i \): Initial Inventory column, table_id: file_0_view_0.
- Parameter \( d_i \): Demand column, table_id: file_0_view_0.