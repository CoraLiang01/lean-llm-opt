Mathematical Model

Let:
- $I$ = set of areas, indexed by $i$, with area names as in ProductName (from products.csv)
- $x_i$ = scale of development per day in area $i$ (decision variable, nonnegative integer)
- $v_i$ = Value of development in area $i$ (from products.csv, column Value)
- $w_i$ = Weight (resource consumption per unit development) in area $i$ (from products.csv, column Weight)
- $C$ = overall development capacity (from capacity.csv, column Capacity)

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

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: decision variable for each $i \in I$ (nonnegative integer)