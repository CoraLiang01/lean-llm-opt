Abstract Optimization Model

Index Sets:
- \( I \): Set of all products classified under 'Baby' (from Product Name in file_0_view_0).

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from Revenue, file_0_view_0).
- \( d_i \): Demand for product \( i \) (from Demand, file_0_view_0).
- \( s_i \): Initial inventory of product \( i \) (from Initial Inventory, file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer (for all \( i \in I \)).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory constraint: \( x_i \leq s_i \)  (for all \( i \in I \))
2. Demand constraint: \( x_i \leq d_i \)  (for all \( i \in I \))
3. Non-negativity and integrality: \( x_i \geq 0 \), integer  (for all \( i \in I \))

Data Mapping:
- Index set \( I \): All records in file_0_view_0 where Product Name contains or starts with 'Baby'
- Parameter \( r_i \): file_0_view_0, column 'Revenue'
- Parameter \( d_i \): file_0_view_0, column 'Demand'
- Parameter \( s_i \): file_0_view_0, column 'Initial Inventory'
- Decision variable \( x_i \): defined for each \( i \in I \)