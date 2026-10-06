ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of all products classified under ‘Baby’ (from EuropeSalesRecords.csv, filtered by Product Name prefix "Baby").

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (EuropeSalesRecords.csv, column: Revenue, table_id: file_0_view_0).
- \( d_i \): Demand quantity for product \( i \) (EuropeSalesRecords.csv, column: Demand, table_id: file_0_view_0).
- \( s_i \): Initial inventory for product \( i \) (EuropeSalesRecords.csv, column: Initial Inventory, table_id: file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), \( x_i \) continuous or integer as appropriate.

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory constraint: \( x_i \leq s_i \) for all \( i \in I \)
2. Demand constraint:  \( x_i \leq d_i \) for all \( i \in I \)
3. Non-negativity:    \( x_i \geq 0 \) for all \( i \in I \)

Data Mapping:
- Index set \( I \), and parameters \( r_i \), \( d_i \), \( s_i \) are sourced from EuropeSalesRecords.csv (table_id: file_0_view_0), with rows filtered where Product Name has prefix "Baby". Columns used: "Product Name" (for index), "Revenue" (for \( r_i \)), "Demand" (for \( d_i \)), "Initial Inventory" (for \( s_i \)).