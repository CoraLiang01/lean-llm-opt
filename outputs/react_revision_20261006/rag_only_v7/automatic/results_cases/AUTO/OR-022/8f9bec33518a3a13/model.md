Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- Let \( I \) be the set of all products in Salesorders.csv whose 'Product Name' contains '27in'.

Parameters (for each \( i \in I \)):
- \( r_i \): Revenue per unit of product \( i \) (from 'Revenue', table_id: file_0_view_0)
- \( d_i \): Demand for product \( i \) (from 'Demand', table_id: file_0_view_0)
- \( s_i \): Initial inventory of product \( i \) (from 'Initial Inventory', table_id: file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i x_i
  \]

Constraints:
1. Inventory and demand fulfillment bounds for each product:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
   \]

Data Mapping:
- Index set \( I \): All records in table_id file_0_view_0 where 'Product Name' contains '27in'
- \( r_i \): 'Revenue' column, table_id file_0_view_0
- \( d_i \): 'Demand' column, table_id file_0_view_0
- \( s_i \): 'Initial Inventory' column, table_id file_0_view_0
- Decision variable \( x_i \): Number of units fulfilled for each \( i \in I \)

This model maximizes total revenue from fulfilling demand for all '27in' products, subject to available inventory and demand limits.