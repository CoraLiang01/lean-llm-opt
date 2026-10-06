ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of all products classified under ‘Baby’. (From EuropeSalesRecords.csv, Product Name with prefix 'Baby')

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (EuropeSalesRecords.csv, Revenue)
- \( d_i \): Demand for product \( i \). (EuropeSalesRecords.csv, Demand)
- \( s_i \): Initial inventory for product \( i \). (EuropeSalesRecords.csv, Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer (if required by context).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory constraint: \( x_i \leq s_i \) for all \( i \in I \)
2. Demand constraint:  \( x_i \leq d_i \) for all \( i \in I \)
3. Non-negativity:    \( x_i \geq 0 \) for all \( i \in I \)

Data Mapping:
- Index set \( I \): All rows in EuropeSalesRecords.csv where Product Name starts with 'Baby' (table_id: file_0_view_0, column: Product Name)
- \( r_i \): EuropeSalesRecords.csv, column: Revenue, table_id: file_0_view_0
- \( d_i \): EuropeSalesRecords.csv, column: Demand, table_id: file_0_view_0
- \( s_i \): EuropeSalesRecords.csv, column: Initial Inventory, table_id: file_0_view_0

This model maximizes total revenue from fulfilling demand for ‘Baby’ products, subject to inventory and demand limits.