ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of car models classified under ‘FDK57’. (Each record with ‘Product Name’ = ‘FDK57’ in table_id: file_0_view_0.)

Parameters:
- \( r_i \): Revenue per unit for car model \( i \). (Source: file_0_view_0, column: Revenue)
- \( s_i \): Initial inventory for car model \( i \). (Source: file_0_view_0, column: Initial Inventory)
- \( d_i \): Demand for car model \( i \). (Source: file_0_view_0, column: Demand)

Decision Variables:
- \( x_i \): Quantity of car model \( i \) to fulfill (integer, \( x_i \geq 0 \)), for all \( i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled sales.)

Constraints:
1. Inventory constraint for each model:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
   (Cannot fulfill more than available inventory.)

2. Demand constraint for each model:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
   (Cannot fulfill more than demand.)

3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_{+} \quad \forall i \in I
   \]
   (Fulfilled quantities are non-negative integers.)

Data Mapping:
- Index set \( I \): All records in table_id: file_0_view_0 with ‘Product Name’ = ‘FDK57’ (CSVQA-applied filter).
- Parameter \( r_i \): file_0_view_0, column: Revenue.
- Parameter \( s_i \): file_0_view_0, column: Initial Inventory.
- Parameter \( d_i \): file_0_view_0, column: Demand.

No additional constraints or selection rules are imposed beyond those specified in the user query and the CSVQA-applied filter.