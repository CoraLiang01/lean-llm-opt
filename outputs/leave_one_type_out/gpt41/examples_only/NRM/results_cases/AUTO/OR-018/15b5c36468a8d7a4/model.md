Abstract Optimization Model

Index Sets:
- \( I \): Set of all products classified as 'Baby' (from [Product Name] in table_id: file_0_view_0).

Parameters:
- \( r_i \): Revenue per unit of product \( i \) ([Revenue], table_id: file_0_view_0).
- \( d_i \): Demand for product \( i \) ([Demand], table_id: file_0_view_0).
- \( s_i \): Initial inventory of product \( i \) ([Initial Inventory], table_id: file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers), for all \( i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory constraint: \( x_i \leq s_i \), for all \( i \in I \).
2. Demand constraint: \( x_i \leq d_i \), for all \( i \in I \).
3. Non-negativity and integrality: \( x_i \geq 0 \) and integer, for all \( i \in I \).

Data Mapping:
- Index set \( I \): All rows in table_id: file_0_view_0 where [Product Name] has prefix 'Baby'.
- Parameter \( r_i \): [Revenue] column, table_id: file_0_view_0.
- Parameter \( d_i \): [Demand] column, table_id: file_0_view_0.
- Parameter \( s_i \): [Initial Inventory] column, table_id: file_0_view_0.