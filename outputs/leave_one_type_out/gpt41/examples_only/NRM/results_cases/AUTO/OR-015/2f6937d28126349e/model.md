Abstract Optimization Model

Index Sets:
- \( I \): Set of all products classified as ‘Aalop’ (from RestaurantSalesreport.csv, Product Name with prefix 'Aalop').

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (RestaurantSalesreport.csv, Revenue).
- \( d_i \): Demand for product \( i \) during the sales horizon (RestaurantSalesreport.csv, Demand).
- \( s_i \): Initial inventory of product \( i \) (RestaurantSalesreport.csv, Initial Inventory).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory constraint: \( x_i \leq s_i \) for all \( i \in I \)
2. Demand constraint:  \( x_i \leq d_i \) for all \( i \in I \)
3. Non-negativity and integrality: \( x_i \geq 0 \), integer for all \( i \in I \)

Data Mapping:
- Table: RestaurantSalesreport.csv
    - Index set \( I \): All rows where Product Name has prefix 'Aalop' (table_id: file_0_view_0, column: Product Name)
    - Parameter \( r_i \): Revenue (table_id: file_0_view_0, column: Revenue)
    - Parameter \( d_i \): Demand (table_id: file_0_view_0, column: Demand)
    - Parameter \( s_i \): Initial Inventory (table_id: file_0_view_0, column: Initial Inventory)