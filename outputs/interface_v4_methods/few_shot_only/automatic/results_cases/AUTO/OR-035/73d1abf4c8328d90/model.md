ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of bread types, indexed by $i$ (from file_1_view_0, column ProductName)

Parameters:
- $p_i$: Expected profit per unit of bread type $i$ (file_1_view_0, column Value, key ProductName)
- $w_i$: Storage space required per unit of bread type $i$ (file_1_view_0, column Weight, key ProductName)
- $C$: Total available storage capacity (file_0_view_0, column Capacity)

Decision Variables:
- $x_i$: Number of units of bread type $i$ to order each day; $x_i \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} p_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Data Mapping:

- $I$ (bread types): file_1_view_0, column ProductName
- $p_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity

All parameters and index sets are to be populated directly from the referenced columns and rows, preserving original file and row order.