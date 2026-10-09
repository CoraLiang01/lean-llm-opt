Mathematical Model (Abstract Formulation)

Index Sets:
- \( I \): Set of vehicle types (indexed by \( i \)), corresponding to all ProductName values in file_1_view_0.

Parameters:
- \( v_i \): Profit from selling one unit of vehicle type \( i \). (file_1_view_0, column: Value)
- \( w_i \): Inventory space consumed by one unit of vehicle type \( i \). (file_1_view_0, column: Weight)
- \( C \): Total available inventory capacity. (file_0_view_0, column: Capacity)

Decision Variables:
- \( x_i \in \mathbb{Z}_+ \): Number of vehicles of type \( i \) to order per day.

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \geq 0 \quad \forall i \in I,\ x_i \in \mathbb{Z}
\]

Data Mapping:
- \( I \): All rows in file_1_view_0, identified by ProductName.
- \( v_i \): file_1_view_0, column Value, for each ProductName \( i \).
- \( w_i \): file_1_view_0, column Weight, for each ProductName \( i \).
- \( C \): file_0_view_0, column Capacity (single value).
- Decision variable \( x_i \): Number of vehicles of type \( i \) to order per day, for each ProductName in file_1_view_0.