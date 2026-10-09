##### Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), corresponding to all ProductName values in file_1_view_0.
- For each area $i \in I$:
    - $v_i$ = Value of developing area $i$ (from file_1_view_0, column Value)
    - $w_i$ = Weight (resource consumption) of developing area $i$ (from file_1_view_0, column Weight)
- $C$ = overall development capacity (from file_0_view_0, column Capacity)
- $x_i$ = scale of development per day in area $i$ (decision variable, nonnegative integer)

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
- $v_i$: file_1_view_0, column Value, for each $i$
- $w_i$: file_1_view_0, column Weight, for each $i$
- $C$: file_0_view_0, column Capacity
- $x_i$: scale of development per day in area $i$ (decision variable, nonnegative integer)