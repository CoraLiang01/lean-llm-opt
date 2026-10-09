##### Mathematical Model

Let:
- $I$ = set of areas, indexed by $i$ (from all ProductName in products.csv)
- $x_i$ = scale of development per day in area $i$ (decision variable, nonnegative integer)
- $v_i$ = development benefit per unit in area $i$ (from Value in products.csv)
- $w_i$ = resource requirement per unit in area $i$ (from Weight in products.csv)
- $C$ = overall development capacity (from Capacity in capacity.csv)

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
- $C$: Capacity column in file_0_view_0 (capacity.csv), row 0
- $x_i$: Decision variable for each $i \in I$ (nonnegative integer)