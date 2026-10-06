ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: Set of areas (indexed by $i$), corresponding to ProductName in file_1_view_0 (products.csv).

Parameters:
- $b_i$: Development benefit for area $i$ (file_1_view_0, column Value, key ProductName).
- $w_i$: Development weight (resource consumption per unit) for area $i$ (file_1_view_0, column Weight, key ProductName).
- $C$: Overall development capacity (file_0_view_0, column Capacity).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Scale of development per day in area $i$.

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

Data Mapping:

- $b_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity

All data and identifiers are preserved as returned.