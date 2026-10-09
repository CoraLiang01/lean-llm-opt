Mathematical Model

Sets:
- $I$: set of areas (indexed by $i$), corresponding to all ProductName values in file_1_view_0.

Parameters:
- $v_i$: benefit coefficient for area $i$ (Value column in file_1_view_0, key: ProductName).
- $w_i$: development unit weight for area $i$ (Weight column in file_1_view_0, key: ProductName).
- $C$: overall development capacity (Capacity column in file_0_view_0).

Decision Variables:
- $x_i$: integer, scale of development in area $i$ per day ($x_i \in \mathbb{Z}_{\geq 0}$).

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

- $I$: All ProductName values in file_1_view_0.
- $v_i$: file_1_view_0, column Value, key ProductName.
- $w_i$: file_1_view_0, column Weight, key ProductName.
- $C$: file_0_view_0, column Capacity.
- $x_i$: integer variable for each $i \in I$.