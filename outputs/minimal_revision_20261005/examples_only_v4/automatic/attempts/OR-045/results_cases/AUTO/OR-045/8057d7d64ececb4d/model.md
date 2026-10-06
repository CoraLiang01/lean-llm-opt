ABSTRACT MATHEMATICAL MODEL

Sets:
- \( I \): Set of produce types, indexed by \( i \). (From file_1_view_0, column ProductName)

Parameters:
- \( v_i \): Benefit per unit of produce \( i \). (From file_1_view_0, column Value)
- \( w_i \): Weight per unit of produce \( i \). (From file_1_view_0, column Weight)
- \( C \): Total inventory capacity (overall weight limit). (From file_0_view_0, column Capacity)

Decision Variables:
- \( x_i \in \mathbb{Z}_+ \): Number of units of produce \( i \) to order daily.

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \geq 0 \quad \text{and integer} \quad \forall i \in I
\]

DATA MAPPING

- Set \( I \): All rows in file_1_view_0, column ProductName.
- Parameter \( v_i \): file_1_view_0, column Value, keyed by ProductName.
- Parameter \( w_i \): file_1_view_0, column Weight, keyed by ProductName.
- Parameter \( C \): file_0_view_0, column Capacity (single value).
- Decision variable \( x_i \): For each ProductName in file_1_view_0.