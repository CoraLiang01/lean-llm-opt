Mathematical Model

Index Sets:
- Let $I$ be the set of areas, with each area identified by ProductName from file_1_view_0.

Parameters:
- $b_i$: development benefit per unit in area $i$ (Value, file_1_view_0)
- $w_i$: resource requirement per unit in area $i$ (Weight, file_1_view_0)
- $C$: overall development capacity (Capacity, file_0_view_0)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: scale of development per day in area $i$

Objective:
\[
\max \sum_{i \in I} b_i x_i
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
- $b_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0
- $x_i$: Decision variable for each ProductName in file_1_view_0

All parameters and index sets are defined directly from the current CSV data.