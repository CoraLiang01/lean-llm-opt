ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of products, indexed by \( i \). (From WomenClothingEcommerceSalesData.csv, column 'Product Name', table_id: file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (WomenClothingEcommerceSalesData.csv, 'Revenue', table_id: file_0_view_0)
- \( d_i \): Demand quantity for product \( i \). (WomenClothingEcommerceSalesData.csv, 'Demand', table_id: file_0_view_0)
- \( s_i \): Initial inventory for product \( i \). (WomenClothingEcommerceSalesData.csv, 'Initial Inventory', table_id: file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled demand.)

Constraints:
1. Inventory and demand fulfillment bounds:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
   \]
   (Cannot fulfill more than available inventory or demand for each product.)

Variable Domains:
- \( x_i \) are integer variables.

Data Mapping:
- Index set \( I \), parameters \( r_i \), \( d_i \), and \( s_i \) are sourced from WomenClothingEcommerceSalesData.csv (table_id: file_0_view_0), columns:
    - 'Product Name' → \( I \)
    - 'Revenue' → \( r_i \)
    - 'Demand' → \( d_i \)
    - 'Initial Inventory' → \( s_i \)
- All 24 records are used; no filters applied.

This model maximizes total revenue by choosing how much demand to fulfill for each product, subject to inventory and demand limits.