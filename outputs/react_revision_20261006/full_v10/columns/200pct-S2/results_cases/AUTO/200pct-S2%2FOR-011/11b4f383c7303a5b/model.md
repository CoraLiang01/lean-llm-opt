Mathematical Model

Index Sets:
- Let $I$ be the set of products, with each $i \in I$ corresponding to a unique ProductName from file_1_view_0.

Parameters:
- $v_i$: Value of product $i$ (from file_1_view_0, column Value)
- $w_i$: Weight of product $i$ (from file_1_view_0, column Weight)
- $C$: Overall stock capacity (from file_0_view_0, column Capacity)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to order each day

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

- $I$: All ProductName in file_1_view_0
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each ProductName in file_1_view_0

All parameters and index sets are defined directly from the current CSV data.