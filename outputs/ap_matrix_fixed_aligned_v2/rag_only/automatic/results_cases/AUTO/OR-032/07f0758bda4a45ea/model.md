Abstract Optimization Model

Index Sets:
- \( I \): Set of all products classified as 'Books', indexed by \( i \).
 (Data Mapping: Each \( i \) corresponds to a row where [Product_Name] has prefix 'Books_' in table_id: file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit for product \( i \).
 (Data Mapping: [Revenue], table_id: file_0_view_0)
- \( s_i \): Initial inventory available for product \( i \).
 (Data Mapping: [Initial Inventory], table_id: file_0_view_0)
- \( d_i \): Demand quantity for product \( i \).
 (Data Mapping: [Demand], table_id: file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill (integer, \( x_i \geq 0 \)).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units of 'Books' products.)

Constraints:
1. Inventory constraint for each product:
\[
x_i \leq s_i \quad \forall i \in I
\]
(The fulfilled quantity cannot exceed available inventory.)

2. Demand constraint for each product:
\[
x_i \leq d_i \quad \forall i \in I
\]
(The fulfilled quantity cannot exceed demand.)

3. Non-negativity and integrality:
\[
x_i \geq 0 \text{ and integer} \quad \forall i \in I
\]

Data Mapping:
- Index set \( I \), parameters \( r_i \), \( s_i \), and \( d_i \) are sourced from table_id: file_0_view_0 ([Product_Name] with prefix 'Books_', [Revenue], [Initial Inventory], [Demand]) in DifferentStoreSales.csv, as returned by the validated filter ([Product_Name] prefix 'Books_').

No additional constraints or modifications are imposed beyond those specified in the user query and the validated data subset.