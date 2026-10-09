Mathematical Optimization Model

Index Sets:
- \( I \): Set of all products with SKU prefix 'ZZ' (from file_0_view_0, column SKU).

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from file_0_view_0, column Revenue).
- \( d_i \): Demand quantity for product \( i \) (from file_0_view_0, column Demand).
- \( s_i \): Initial inventory of product \( i \) (from file_0_view_0, column Initial Inventory).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

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
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set \( I \): All SKUs in file_0_view_0, column SKU, where SKU starts with 'ZZ'.
- Parameter \( r_i \): file_0_view_0, column Revenue.
- Parameter \( d_i \): file_0_view_0, column Demand.
- Parameter \( s_i \): file_0_view_0, column Initial Inventory.