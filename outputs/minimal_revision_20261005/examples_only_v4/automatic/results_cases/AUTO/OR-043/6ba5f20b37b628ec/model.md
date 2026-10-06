ABSTRACT MATHEMATICAL MODEL

Sets:
- \( I \): Set of drug products, indexed by \( i \) (corresponds to all ProductName in products.csv).

Parameters:
- \( v_i \): Benefit per unit of drug \( i \). [From file_1_view_0, column Value]
- \( w_i \): Stock weight per unit of drug \( i \). [From file_1_view_0, column Weight]
- \( C \): Overall stock capacity for the pharmacy chain. [From file_0_view_0, column Capacity]

Decision Variables:
- \( x_i \): Number of units of drug \( i \) to order each day. (integer, \( x_i \geq 0 \))

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in I
\]

DATA MAPPING

- Index set \( I \): All rows in file_1_view_0, column ProductName.
- Parameter \( v_i \): file_1_view_0, column Value, keyed by ProductName.
- Parameter \( w_i \): file_1_view_0, column Weight, keyed by ProductName.
- Parameter \( C \): file_0_view_0, column Capacity (single value).
- Decision variable \( x_i \): For each ProductName in file_1_view_0.