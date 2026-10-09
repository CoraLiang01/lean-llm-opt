##### Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), with area names from file_1_view_0.ProductName
- $x_i$ = scale of development per day in area $i$ (decision variable, nonnegative integer)
- $v_i$ = development benefit per unit in area $i$ (from file_1_view_0.Value)
- $w_i$ = resource usage per unit in area $i$ (from file_1_view_0.Weight)
- $C$ = overall development capacity (from file_0_view_0.Capacity)

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

- $I$: file_1_view_0.ProductName
- $v_i$: file_1_view_0.Value, matched by ProductName $i$
- $w_i$: file_1_view_0.Weight, matched by ProductName $i$
- $C$: file_0_view_0.Capacity
- $x_i$: scale of development per day in area $i$ (decision variable, nonnegative integer)