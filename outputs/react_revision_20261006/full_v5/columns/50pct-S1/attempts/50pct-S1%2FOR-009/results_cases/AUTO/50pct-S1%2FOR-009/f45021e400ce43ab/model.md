#### Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), with area names from column ProductName in file_1_view_0.
- For each $i \in I$:
    - $v_i$ = Value of developing area $i$ (from column Value in file_1_view_0)
    - $w_i$ = Resource requirement ("Weight") for area $i$ (from column Weight in file_1_view_0)
- $C$ = overall development capacity (from column Capacity in file_0_view_0)
- Decision variables: $x_i \in \mathbb{Z}_{\geq 0}$, the scale of development per day in area $i$

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

---

#### Data Mapping

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, matched by ProductName
- $w_i$: Weight column in file_1_view_0, matched by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv), row 0
- $x_i$: Decision variable for each $i \in I$ (area/ProductName)