Abstract Optimization Model

Index Sets:
- \( I \): Set of products, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit of product \( i \).  
  (Data: OnlineSalesDataset.csv, column 'Revenue')
- \( d_i \): Deterministic demand for product \( i \) over the sales horizon.  
  (Data: OnlineSalesDataset.csv, column 'Demand')
- \( s_i \): Initial inventory available for product \( i \).  
  (Data: OnlineSalesDataset.csv, column 'Initial Inventory')

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill for customer purchases.  
  (Domain: integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \) for all \( i \in I \))

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from all products.)

Constraints:
1. Inventory and demand fulfillment:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
   \]
   (Cannot fulfill more than available inventory or demand for any product.)

Data Mapping:
- Product index set \( I \): All rows in OnlineSalesDataset.csv (table_id: file_0_view_0), column 'Product Name'
- Revenue per unit \( r_i \): OnlineSalesDataset.csv (table_id: file_0_view_0), column 'Revenue'
- Demand \( d_i \): OnlineSalesDataset.csv (table_id: file_0_view_0), column 'Demand'
- Initial Inventory \( s_i \): OnlineSalesDataset.csv (table_id: file_0_view_0), column 'Initial Inventory'

No filters were applied; all products in the table are included. All parameter values are to be taken directly from the specified columns for each product.