ABSTRACT MATHEMATICAL MODEL

Sets:
- \( I \): Set of drug products, indexed by \( i \) (corresponds to all ProductName in products.csv).

Parameters:
- \( v_i \): Benefit per unit of drug \( i \). (from Value column in products.csv)
- \( w_i \): Stock weight per unit of drug \( i \). (from Weight column in products.csv)
- \( C \): Overall stock capacity. (from Capacity column in capacity.csv)

Decision Variables:
- \( x_i \): Number of units of drug \( i \) to order each day. (\( x_i \in \mathbb{Z}_+, \) i.e., nonnegative integers)

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \geq 0 \quad \forall i \in I,\quad x_i \in \mathbb{Z}
\]

DATA MAPPING

- \( I \): All rows in file_1_view_0 (products.csv), indexed by ProductName.
- \( v_i \): file_1_view_0, column Value, for each ProductName.
- \( w_i \): file_1_view_0, column Weight, for each ProductName.
- \( C \): file_0_view_0, column Capacity, row 0.
- Decision variable \( x_i \) is defined for each ProductName in file_1_view_0.