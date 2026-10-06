ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of products classified under ‘id999’. (From OnlineRetailSalesDataset.csv, table_id: file_0_view_0, column: id_number)

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (table_id: file_0_view_0, column: Revenue)
- \( d_i \): Demand for product \( i \) during the sales horizon. (table_id: file_0_view_0, column: Demand)
- \( s_i \): Initial inventory of product \( i \). (table_id: file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers), for all \( i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory constraint: \( x_i \leq s_i \), for all \( i \in I \)
2. Demand constraint:  \( x_i \leq d_i \), for all \( i \in I \)
3. Non-negativity and integrality: \( x_i \in \mathbb{Z}_+ \), for all \( i \in I \)

Data Mapping:
- Index set \( I \), and parameters \( r_i \), \( d_i \), \( s_i \) are sourced from OnlineRetailSalesDataset.csv (table_id: file_0_view_0), columns: id_number, Revenue, Demand, Initial Inventory, filtered for id_number = 'id999'.