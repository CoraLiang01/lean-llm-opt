ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of all products classified under 'Baby'.
 (Data: All rows in Salesdata.csv where Product Name has prefix 'Baby'; table_id: file_0_view_0, column: Product Name)

Parameters:
- \( r_i \): Revenue per unit of product \( i \).
 (Data: Salesdata.csv, table_id: file_0_view_0, column: Revenue)
- \( d_i \): Demand for product \( i \).
 (Data: Salesdata.csv, table_id: file_0_view_0, column: Demand)
- \( s_i \): Initial inventory for product \( i \).
 (Data: Salesdata.csv, table_id: file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill.
 Domain: Integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \), \( \forall i \in I \)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Demand and Inventory Fulfillment:
 \[
 0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
 \]
 (Alternatively, as two constraints:)
 \[
 x_i \leq d_i \quad \forall i \in I
 \]
 \[
 x_i \leq s_i \quad \forall i \in I
 \]
2. Integrality:
 \[
 x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
 \]

Data Mapping:
- Index set \( I \): All rows in Salesdata.csv where Product Name has prefix 'Baby' (table_id: file_0_view_0, column: Product Name)
- Parameter \( r_i \): Salesdata.csv, table_id: file_0_view_0, column: Revenue
- Parameter \( d_i \): Salesdata.csv, table_id: file_0_view_0, column: Demand
- Parameter \( s_i \): Salesdata.csv, table_id: file_0_view_0, column: Initial Inventory

No data values or record counts are included per instructions.