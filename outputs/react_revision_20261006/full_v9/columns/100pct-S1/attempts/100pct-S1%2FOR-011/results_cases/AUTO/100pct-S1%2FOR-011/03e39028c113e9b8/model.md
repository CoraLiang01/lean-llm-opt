Mathematical Model

Index Sets:
- $I$: Set of products, indexed by $i$ (from ProductName in file_1_view_0).

Parameters:
- $v_i$: Value (benefit) per unit of product $i$ (from Value in file_1_view_0).
- $w_i$: Weight (stock space required) per unit of product $i$ (from Weight in file_1_view_0).
- $C$: Overall stock capacity (from Capacity in file_0_view_0).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to order each day.

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Data Mapping

- $I$: All ProductName in file_1_view_0.
- $v_i$: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $C$: file_0_view_0, column Capacity.
- $x_i$: Decision variable for each $i \in I$.