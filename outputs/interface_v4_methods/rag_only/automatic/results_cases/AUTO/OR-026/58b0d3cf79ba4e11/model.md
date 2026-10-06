ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of all products classified under 'Fashion'  
  (Source: SupermarketSales.csv, table_id: file_0_view_0, column: Product Name, filtered by prefix 'Fashion')

Parameters:
- \( r_i \): Revenue per unit of product \( i \)  
  (Source: SupermarketSales.csv, table_id: file_0_view_0, column: Revenue)
- \( s_i \): Initial inventory of product \( i \)  
  (Source: SupermarketSales.csv, table_id: file_0_view_0, column: Initial Inventory)
- \( d_i \): Demand for product \( i \)  
  (Source: SupermarketSales.csv, table_id: file_0_view_0, column: Demand)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill  
  Domain: Integer, \( 0 \leq x_i \leq \min(s_i, d_i) \), for all \( i \in I \)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory and demand fulfillment:
   \[
   0 \leq x_i \leq \min(s_i, d_i) \quad \forall i \in I
   \]
   (Each product's fulfilled quantity cannot exceed its available inventory or demand.)

Data Mapping:
- Index set \( I \): All rows in SupermarketSales.csv (table_id: file_0_view_0) where 'Product Name' starts with 'Fashion'
- Parameter \( r_i \): 'Revenue' column, table_id: file_0_view_0
- Parameter \( s_i \): 'Initial Inventory' column, table_id: file_0_view_0
- Parameter \( d_i \): 'Demand' column, table_id: file_0_view_0

No literal data values or record counts are included. All mappings are symbolic and reference the exact table and column names.