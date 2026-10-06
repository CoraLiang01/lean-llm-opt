ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of products/categories, indexed by \( i \).
 (Data source: RetailSalesDataset.csv, table_id: file_0_view_0, column: Product Name)

Parameters:
- \( r_i \): Revenue per unit for product \( i \).
 (Data source: RetailSalesDataset.csv, table_id: file_0_view_0, column: Revenue)
- \( d_i \): Demand quantity for product \( i \).
 (Data source: RetailSalesDataset.csv, table_id: file_0_view_0, column: Demand)
- \( s_i \): Initial inventory available for product \( i \).
 (Data source: RetailSalesDataset.csv, table_id: file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Quantity of product \( i \) to fulfill (allocate to demand), \( x_i \geq 0 \), continuous or integer as appropriate.

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from all fulfilled quantities.)

Constraints:
1. Inventory and demand fulfillment limits for each product:
\[
0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
\]
(Do not fulfill more than available inventory or demand.)

Data Mapping:
- Index set \( I \): file_0_view_0, column Product Name
- Parameter \( r_i \): file_0_view_0, column Revenue
- Parameter \( d_i \): file_0_view_0, column Demand
- Parameter \( s_i \): file_0_view_0, column Initial Inventory

This model symbolically represents the optimal allocation of merchandise to maximize revenue, subject to inventory and demand constraints, with all data sources explicitly mapped.