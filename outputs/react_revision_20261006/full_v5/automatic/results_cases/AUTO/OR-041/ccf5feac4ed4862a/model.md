##### Mathematical Model

Let:
- $I$ = set of areas (from products.csv, column ProductName)
- For each $i \in I$:
    - $v_i$ = Value of developing area $i$ (products.csv, column Value)
    - $w_i$ = Resource required per unit scale in area $i$ (products.csv, column Weight)
    - $x_i$ = scale of development per day in area $i$ (decision variable, nonnegative integer)
- $C$ = overall development capacity (capacity.csv, column Capacity)

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

- $I$: All ProductName values in file_1_view_0 (products.csv, column ProductName)
- $v_i$: file_1_view_0, column Value, for each $i$
- $w_i$: file_1_view_0, column Weight, for each $i$
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (nonnegative integer)