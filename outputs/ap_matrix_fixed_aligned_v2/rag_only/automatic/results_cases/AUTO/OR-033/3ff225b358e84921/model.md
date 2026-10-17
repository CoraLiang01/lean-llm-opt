ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of products classified under ‘Baby’ in the European market.
  (Data Mapping: All records in EuropeSalesRecords.csv where Product Name has prefix 'Baby', table_id: file_0_view_0, column: Product Name)

Parameters:
- \( r_i \): Revenue per unit of product \( i \).
  (Data Mapping: EuropeSalesRecords.csv, table_id: file_0_view_0, column: Revenue)
- \( d_i \): Demand quantity for product \( i \).
  (Data Mapping: EuropeSalesRecords.csv, table_id: file_0_view_0, column: Demand)
- \( s_i \): Initial inventory for product \( i \).
  (Data Mapping: EuropeSalesRecords.csv, table_id: file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill (integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \), \( \forall i \in I \)).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units.)

Constraints:
1. Inventory and Demand Fulfillment:
  \[
  0 \leq x_i \leq \min\{d_i, s_i\} \qquad \forall i \in I
  \]
  (Do not fulfill more than available inventory or demand.)

Variable Domains:
- \( x_i \) are integer variables.

Data Mapping:
- Index set \( I \), and all parameters \( r_i, d_i, s_i \) are sourced from EuropeSalesRecords.csv (table_id: file_0_view_0), using only records where Product Name has prefix 'Baby' (validated filter).
- Revenue: column 'Revenue'
- Demand: column 'Demand'
- Initial Inventory: column 'Initial Inventory'
- Product identifier: column 'Product Name'

No additional constraints or requirements are imposed beyond those specified in the user query and the validated data subset.