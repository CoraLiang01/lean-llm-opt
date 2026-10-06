ABSTRACT Mathematical Optimization Model

Index Sets:
- \( I \): Set of products with category prefix 'ZZ', indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit of product \( i \). Source: RetailStoreSalesTransactions(ScannerData).csv, column [Revenue], table_id: file_0_view_0.
- \( s_i \): Initial inventory of product \( i \). Source: RetailStoreSalesTransactions(ScannerData).csv, column [Initial Inventory], table_id: file_0_view_0.
- \( d_i \): Demand for product \( i \). Source: RetailStoreSalesTransactions(ScannerData).csv, column [Demand], table_id: file_0_view_0.

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer (if only whole units can be fulfilled).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units of 'ZZ' products.)

Constraints:
1. Inventory constraint: \( x_i \leq s_i \) for all \( i \in I \)
2. Demand constraint:  \( x_i \leq d_i \) for all \( i \in I \)
3. Non-negativity:    \( x_i \geq 0 \) for all \( i \in I \)
4. (Optional) Integrality: \( x_i \) integer for all \( i \in I \) (if required by business rules)

Data Mapping:
- Index set \( I \): All rows in RetailStoreSalesTransactions(ScannerData).csv where [SKU] has prefix 'ZZ' (table_id: file_0_view_0).
- \( r_i \): [Revenue], table_id: file_0_view_0.
- \( s_i \): [Initial Inventory], table_id: file_0_view_0.
- \( d_i \): [Demand], table_id: file_0_view_0.