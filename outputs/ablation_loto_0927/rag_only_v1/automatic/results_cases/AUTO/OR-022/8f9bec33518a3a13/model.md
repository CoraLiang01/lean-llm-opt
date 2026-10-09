ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let \( I \) be the set of all products classified under ‘27in’. (From Salesorders.csv, Product Name with prefix '27in')

Parameters:
- \( r_i \): Revenue per unit of product \( i \in I \). (Salesorders.csv, Revenue, table_id: file_0_view_0)
- \( s_i \): Initial inventory of product \( i \in I \). (Salesorders.csv, Initial Inventory, table_id: file_0_view_0)
- \( d_i \): Demand for product \( i \in I \). (Salesorders.csv, Demand, table_id: file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \in I \) to fulfill. (\( x_i \geq 0 \), integer or continuous as appropriate)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory constraint: \( x_i \leq s_i \) for all \( i \in I \)
2. Demand constraint:  \( x_i \leq d_i \) for all \( i \in I \)
3. Non-negativity:    \( x_i \geq 0 \) for all \( i \in I \)

Data Mapping:
- Index set \( I \): All rows in Salesorders.csv (table_id: file_0_view_0) where Product Name starts with '27in'
- \( r_i \): Salesorders.csv, column 'Revenue', table_id: file_0_view_0
- \( s_i \): Salesorders.csv, column 'Initial Inventory', table_id: file_0_view_0
- \( d_i \): Salesorders.csv, column 'Demand', table_id: file_0_view_0