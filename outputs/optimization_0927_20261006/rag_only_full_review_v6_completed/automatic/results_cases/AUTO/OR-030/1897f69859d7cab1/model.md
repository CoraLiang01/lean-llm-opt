Abstract Optimization Model

Index Sets:
- \( I \): Set of car models classified under ‘FDK57’ (indexed by \( i \)), as returned by the filtered records.

Parameters:
- \( r_i \): Revenue per unit for car model \( i \) (from column Revenue, table_id: file_0_view_0).
- \( s_i \): Initial inventory for car model \( i \) (from column Initial Inventory, table_id: file_0_view_0).
- \( d_i \): Demand for car model \( i \) (from column Demand, table_id: file_0_view_0).

Decision Variables:
- \( x_i \): Quantity of car model \( i \) to fulfill, integer, \( x_i \geq 0 \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory constraint for each car model:
\[
x_i \leq s_i \quad \forall i \in I
\]
2. Demand constraint for each car model:
\[
x_i \leq d_i \quad \forall i \in I
\]
3. Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in I
\]

Data Mapping:
- Index set \( I \): All records in table_id: file_0_view_0 where Product Name = ‘FDK57’.
- \( r_i \): Revenue (column Revenue, table_id: file_0_view_0).
- \( s_i \): Initial Inventory (column Initial Inventory, table_id: file_0_view_0).
- \( d_i \): Demand (column Demand, table_id: file_0_view_0).

All data is mapped directly from the filtered records with [Product Name] = ‘FDK57’ as returned by CSVQA. No additional filtering or aggregation is performed.