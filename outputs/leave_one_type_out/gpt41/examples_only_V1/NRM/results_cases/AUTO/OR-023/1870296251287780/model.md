ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of products classified under ‘ELE-S’, indexed by \( i \).
  (Source: SalesStoreoverview.csv, table_id: file_0_view_0, column: Product_Reference, prefix 'ELE-S')

Parameters:
- \( r_i \): Revenue per unit of product \( i \).
  (Source: SalesStoreoverview.csv, table_id: file_0_view_0, column: Revenue)
- \( s_i \): Initial inventory of product \( i \).
  (Source: SalesStoreoverview.csv, table_id: file_0_view_0, column: Initial Inventory)
- \( d_i \): Demand for product \( i \).
  (Source: SalesStoreoverview.csv, table_id: file_0_view_0, column: Demand)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill.
  Domain: Integer, \( 0 \leq x_i \leq \min(s_i, d_i) \), for all \( i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units.)

Constraints:
1. Inventory constraint:
\[
x_i \leq s_i \quad \forall i \in I
\]
2. Demand constraint:
\[
x_i \leq d_i \quad \forall i \in I
\]
3. Non-negativity and integrality:
\[
x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
\]

Data Mapping:
- Product set \( I \): SalesStoreoverview.csv, table_id: file_0_view_0, column: Product_Reference (prefix 'ELE-S')
- Revenue \( r_i \): SalesStoreoverview.csv, table_id: file_0_view_0, column: Revenue
- Initial Inventory \( s_i \): SalesStoreoverview.csv, table_id: file_0_view_0, column: Initial Inventory
- Demand \( d_i \): SalesStoreoverview.csv, table_id: file_0_view_0, column: Demand