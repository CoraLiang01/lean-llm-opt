##### Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), with each area identified by its ProductName from products.csv.
- $x_i$ = integer variable: daily scale of development in area $i$.
- $v_i$ = benefit coefficient for area $i$ (Value column).
- $w_i$ = resource consumption per unit in area $i$ (Weight column).
- $C$ = overall development capacity (Capacity column from capacity.csv).

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

##### Data Mapping

- $I$: All ProductName values in file_1_view_0 (products.csv).
- $v_i$: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $C$: file_0_view_0, column Capacity.
- $x_i$: decision variable for each $i \in I$.