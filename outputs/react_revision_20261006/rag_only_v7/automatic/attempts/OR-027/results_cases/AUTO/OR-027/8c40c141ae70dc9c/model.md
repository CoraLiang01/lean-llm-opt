Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- \( I \): Set of all products classified under ‘Organ’ (from Sub Category in table_id file_0_view_0).

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from Revenue, file_0_view_0, column "Revenue").
- \( d_i \): Demand quantity for product \( i \) (from Demand, file_0_view_0, column "Demand").
- \( s_i \): Initial inventory for product \( i \) (from Initial Inventory, file_0_view_0, column "Initial Inventory").

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer (for all \( i \in I \)).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units.)

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set \( I \): All products where Sub Category starts with "Organ" in table_id file_0_view_0, column "Sub Category".
- Parameter \( r_i \): Revenue per unit from table_id file_0_view_0, column "Revenue".
- Parameter \( d_i \): Demand from table_id file_0_view_0, column "Demand".
- Parameter \( s_i \): Initial Inventory from table_id file_0_view_0, column "Initial Inventory".