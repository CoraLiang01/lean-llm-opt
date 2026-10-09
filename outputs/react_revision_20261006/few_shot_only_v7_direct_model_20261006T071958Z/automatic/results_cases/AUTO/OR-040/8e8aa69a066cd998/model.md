ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of areas (indexed by $i$), from file_1_view_0 ProductName.

Parameters:
- $v_i$: benefit coefficient for area $i$ (file_1_view_0, Value, keyed by ProductName)
- $w_i$: development unit weight for area $i$ (file_1_view_0, Weight, keyed by ProductName)
- $C$: total development capacity (file_0_view_0, Capacity)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: integer scale of development in area $i$ per day

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

- $I$: file_1_view_0 ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: integer variable for each $i \in I$