Abstract Optimization Model

Index Sets:
- \( I \): Set of all products classified under 'Baby' (indexed by \( i \)).

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (EuropeSalesRecords.csv, column: Revenue)
- \( d_i \): Demand quantity for product \( i \). (EuropeSalesRecords.csv, column: Demand)
- \( s_i \): Initial inventory for product \( i \). (EuropeSalesRecords.csv, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer or continuous as appropriate.

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in I
   \]

Data Mapping:
- Table: EuropeSalesRecords.csv (table_id: file_0_view_0)
    - Product index set \( I \): All rows where 'Product Name' has prefix 'Baby'
    - Revenue per unit (\( r_i \)): column 'Revenue'
    - Demand (\( d_i \)): column 'Demand'
    - Initial Inventory (\( s_i \)): column 'Initial Inventory'