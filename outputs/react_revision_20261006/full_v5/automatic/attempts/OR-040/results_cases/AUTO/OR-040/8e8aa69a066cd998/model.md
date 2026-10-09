Mathematical Model

Sets:
- $I$: set of areas (indexed by $i$), corresponding to all ProductName values in products.csv.

Parameters:
- $v_i$: benefit coefficient for area $i$ (from Value column in products.csv).
- $w_i$: development unit weight for area $i$ (from Weight column in products.csv).
- $C$: overall development capacity (from Capacity column in capacity.csv).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: integer scale of development in area $i$ per day.

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

- $I$: All ProductName values in file_1_view_0 (products.csv, column ProductName)
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: decision variable for each $i \in I$