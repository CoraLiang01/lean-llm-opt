Mathematical Optimization Model (Abstract Formulation)

Sets:
- \( I \): Set of dairy products, indexed by \( i \).  
  (Source: DairyGoodsSalesDataset.csv, column Full_Product_Name, table_id: file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of product \( i \).  
  (Source: DairyGoodsSalesDataset.csv, column Revenue, table_id: file_0_view_0)
- \( d_i \): Demand quantity for product \( i \).  
  (Source: DairyGoodsSalesDataset.csv, column Demand, table_id: file_0_view_0)
- \( s_i \): Initial inventory available for product \( i \).  
  (Source: DairyGoodsSalesDataset.csv, column Initial Inventory, table_id: file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill (integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \)).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled orders.)

Constraints:
1. Inventory and demand fulfillment bounds:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
   \]
   (Cannot fulfill more than available inventory or demand for any product.)

Variable Domains:
- \( x_i \) are integer variables for all \( i \in I \).

---

Data Mapping

- Set \( I \): All values in Full_Product_Name from table_id file_0_view_0
- Parameter \( r_i \): Revenue from column Revenue, table_id file_0_view_0
- Parameter \( d_i \): Demand from column Demand, table_id file_0_view_0
- Parameter \( s_i \): Initial Inventory from column Initial Inventory, table_id file_0_view_0