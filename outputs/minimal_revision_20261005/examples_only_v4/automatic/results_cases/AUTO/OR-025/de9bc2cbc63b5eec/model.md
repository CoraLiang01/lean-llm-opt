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
(Maximize total revenue from fulfilled units.)

Constraints:
1. Inventory and demand fulfillment:
\[
0 \leq x_i \leq \min\{d_i, s_i\} \qquad \forall i \in I
\]
(Units fulfilled cannot exceed either demand or available inventory.)

Data Mapping:
- Index set \( I \): All records in file_0_view_0, column Product Name (filtered by prefix "TABLET")
- Parameter \( r_i \): file_0_view_0, column Revenue
- Parameter \( d_i \): file_0_view_0, column Demand
- Parameter \( s_i \): file_0_view_0, column Initial Inventory

No additional constraints or data sources are used. All parameters are directly mapped from the specified columns.