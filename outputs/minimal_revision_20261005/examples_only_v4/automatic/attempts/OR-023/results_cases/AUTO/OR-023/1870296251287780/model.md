Mathematical Optimization Model

Index Sets:
- \( I \): Set of products classified under ‘ELE-S’, indexed by \( i \).
  (Data: SalesStoreoverview.csv, table_id: file_0_view_0, column: Product_Reference, filtered by prefix ‘ELE-S’)

Parameters:
- \( r_i \): Revenue per unit of product \( i \).
  (Data: file_0_view_0, column: Revenue)
- \( d_i \): Demand quantity for product \( i \).
  (Data: file_0_view_0, column: Demand)
- \( s_i \): Initial inventory available for product \( i \).
  (Data: file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill (integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \)).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units.)

Constraints:
1. Demand fulfillment constraint:
\[
x_i \leq d_i \quad \forall i \in I
\]
2. Inventory availability constraint:
\[
x_i \leq s_i \quad \forall i \in I
\]
3. Non-negativity and integrality:
\[
x_i \geq 0,\quad x_i \in \mathbb{Z} \quad \forall i \in I
\]

Data Mapping

- Index set \( I \): file_0_view_0, column Product_Reference (filtered by prefix ‘ELE-S’)
- Parameter \( r_i \): file_0_view_0, column Revenue
- Parameter \( d_i \): file_0_view_0, column Demand
- Parameter \( s_i \): file_0_view_0, column Initial Inventory