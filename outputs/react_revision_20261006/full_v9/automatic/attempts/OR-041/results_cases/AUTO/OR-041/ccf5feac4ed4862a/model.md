#### Mathematical Model

Let:
- $I$ = set of areas (from file_1_view_0, column ProductName)
- For each $i \in I$:
    - $v_i$ = Value of developing area $i$ (file_1_view_0, column Value)
    - $w_i$ = Resource required to develop area $i$ (file_1_view_0, column Weight)
- $C$ = overall development capacity (file_0_view_0, column Capacity)
- Decision variable: $x_i$ = scale of development per day in area $i$ (nonnegative integer)

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

#### Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv, column ProductName)
- $v_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (area)