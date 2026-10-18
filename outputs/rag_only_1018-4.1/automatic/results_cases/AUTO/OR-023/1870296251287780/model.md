ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of products classified under 'ELE-S'.
  (Data Mapping: All products where [Product_Reference] starts with 'ELE-S' in SalesStoreoverview.csv, table_id: file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of product \( i \).
  (Data Mapping: [Revenue], table_id: file_0_view_0)
- \( s_i \): Initial inventory of product \( i \).
  (Data Mapping: [Initial Inventory], table_id: file_0_view_0)
- \( d_i \): Demand for product \( i \).
  (Data Mapping: [Demand], table_id: file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( x_i \geq 0 \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory constraint for each product:
\[
x_i \leq s_i \quad \forall i \in I
\]
2. Demand constraint for each product:
\[
x_i \leq d_i \quad \forall i \in I
\]
3. Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Data Mapping:
- Index set \( I \), and parameters \( r_i \), \( s_i \), \( d_i \) are sourced from SalesStoreoverview.csv (table_id: file_0_view_0), filtered by [Product_Reference] prefix 'ELE-S'.
- [Revenue] → \( r_i \)
- [Initial Inventory] → \( s_i \)
- [Demand] → \( d_i \)
- [Product_Reference] identifies each \( i \in I \).

No additional constraints or modifications are imposed beyond those specified in the user query and the validated data filter.