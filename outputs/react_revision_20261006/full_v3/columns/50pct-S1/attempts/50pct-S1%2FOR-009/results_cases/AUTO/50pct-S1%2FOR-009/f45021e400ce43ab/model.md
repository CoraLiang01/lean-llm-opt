##### Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), with area names from products.csv ProductName.
- $x_i$ = scale of development per day in area $i$ (decision variable, nonnegative integer).
- $v_i$ = development benefit per unit in area $i$ (from products.csv Value).
- $w_i$ = development capacity required per unit in area $i$ (from products.csv Weight).
- $C$ = overall development capacity (from capacity.csv Capacity).

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

- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, matched by ProductName
- $w_i$: Weight column in file_1_view_0, matched by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv)
- $x_i$: Decision variable for each $i \in I$ (area)