ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of products, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit of product \( i \).  
  Source: WomenClothingEcommerceSalesData.csv, column 'Revenue', table_id: file_0_view_0
- \( d_i \): Demand quantity for product \( i \).  
  Source: WomenClothingEcommerceSalesData.csv, column 'Demand', table_id: file_0_view_0
- \( s_i \): Initial inventory for product \( i \).  
  Source: WomenClothingEcommerceSalesData.csv, column 'Initial Inventory', table_id: file_0_view_0

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill (integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \)).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled demand.)

Constraints:
1. Inventory constraint:  
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
   (Cannot fulfill more than available inventory.)

2. Demand constraint:  
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
   (Cannot fulfill more than demand.)

3. Non-negativity and integrality:  
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set \( I \): All unique values in 'Product Name' (WomenClothingEcommerceSalesData.csv, table_id: file_0_view_0)
- Parameter \( r_i \): 'Revenue' column (WomenClothingEcommerceSalesData.csv, table_id: file_0_view_0)
- Parameter \( d_i \): 'Demand' column (WomenClothingEcommerceSalesData.csv, table_id: file_0_view_0)
- Parameter \( s_i \): 'Initial Inventory' column (WomenClothingEcommerceSalesData.csv, table_id: file_0_view_0)