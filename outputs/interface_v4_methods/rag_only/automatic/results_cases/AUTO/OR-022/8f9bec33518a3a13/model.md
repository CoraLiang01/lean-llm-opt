ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let \( I \) be the set of all products classified under ‘27in’. (Each product \( i \in I \) corresponds to a unique ‘Product Name’ containing ‘27in’ in the Salesorders.csv table.)

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (‘Revenue’ column, table_id: file_0_view_0)
- \( s_i \): Initial inventory of product \( i \) (‘Initial Inventory’ column, table_id: file_0_view_0)
- \( d_i \): Demand for product \( i \) (‘Demand’ column, table_id: file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers), for all \( i \in I \)

Objective:
\[
\text{Maximize} \quad Z = \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory constraint: \( x_i \leq s_i \) for all \( i \in I \)
2. Demand fulfillment:  \( x_i \leq d_i \) for all \( i \in I \)
3. Non-negativity and integrality: \( x_i \geq 0 \), integer, for all \( i \in I \)

Data Mapping:
- Index set \( I \): All rows in table_id: file_0_view_0 (‘Salesorders.csv’) where ‘Product Name’ contains ‘27in’
- Parameter \( r_i \): ‘Revenue’ column, table_id: file_0_view_0
- Parameter \( s_i \): ‘Initial Inventory’ column, table_id: file_0_view_0
- Parameter \( d_i \): ‘Demand’ column, table_id: file_0_view_0

This model maximizes total revenue from fulfilling demand for ‘27in’ products, subject to inventory and demand constraints, using the provided data sources.