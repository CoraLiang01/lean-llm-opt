ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of all products classified as ‘FAUX’ (from ZARASales.csv, [Product Name] where [Product Name] contains 'FAUX').

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from ZARASales.csv, [Revenue], table_id: file_0_view_0).
- \( s_i \): Initial inventory of product \( i \) (from ZARASales.csv, [Initial Inventory], table_id: file_0_view_0).
- \( d_i \): Demand for product \( i \) (from ZARASales.csv, [Demand], table_id: file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers), for all \( i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory constraint: \( x_i \leq s_i \) for all \( i \in I \)
2. Demand constraint:  \( x_i \leq d_i \) for all \( i \in I \)
3. Non-negativity and integrality: \( x_i \geq 0 \), \( x_i \) integer, for all \( i \in I \)

Data Mapping:
- Index set \( I \): All rows in ZARASales.csv (table_id: file_0_view_0) where [Product Name] contains 'FAUX'
- Parameter \( r_i \): [Revenue] column, table_id: file_0_view_0
- Parameter \( s_i \): [Initial Inventory] column, table_id: file_0_view_0
- Parameter \( d_i \): [Demand] column, table_id: file_0_view_0

No data values or record counts are included; all sets and parameters are defined symbolically and mapped to their exact sources.