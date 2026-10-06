Mathematical Optimization Model

Index Sets:
- \( I \): Set of all TABLET smartphone models, indexed by \( i \).
  (Source: SmartphoneRetailOutletSalesData.csv, table_id: file_0_view_0, column: Product Name, filtered by prefix "TABLET")

Parameters:
- \( r_i \): Revenue per unit for model \( i \).
  (Source: file_0_view_0, column: Revenue)
- \( d_i \): Demand quantity for model \( i \).
  (Source: file_0_view_0, column: Demand)
- \( s_i \): Initial inventory for model \( i \).
  (Source: file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of model \( i \) to fulfill.
  Domain: Integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \), for all \( i \in I \)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units)

Constraints:
1. Demand and Inventory Fulfillment:
\[
0 \leq x_i \leq \min\{d_i, s_i\} \qquad \forall i \in I
\]
 (Do not fulfill more than available inventory or demand for any model)

Data Mapping:
- Index set \( I \): All records in file_0_view_0 where Product Name starts with "TABLET"
- \( r_i \): file_0_view_0, column: Revenue
- \( d_i \): file_0_view_0, column: Demand
- \( s_i \): file_0_view_0, column: Initial Inventory

No additional constraints or data sources are used. All parameters are bound directly to the specified columns in the provided table.