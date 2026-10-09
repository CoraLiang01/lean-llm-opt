ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of areas (indexed by $i$), corresponding to all ProductName values in file_1_view_0.

Parameters:
- $v_i$: development benefit per unit in area $i$ (Value column in file_1_view_0, mapped by ProductName).
- $w_i$: development capacity required per unit in area $i$ (Weight column in file_1_view_0, mapped by ProductName).
- $C$: total available development capacity (Capacity column in file_0_view_0).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of real estate units to develop in area $i$ per day.

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

DATA MAPPING

- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, mapped by ProductName
- $w_i$: Weight column in file_1_view_0, mapped by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv)