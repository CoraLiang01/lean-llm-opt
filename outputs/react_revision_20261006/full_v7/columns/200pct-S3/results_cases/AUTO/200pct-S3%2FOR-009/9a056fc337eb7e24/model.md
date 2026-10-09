#### Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), with area names as in file_1_view_0.ProductName
- $x_i$ = scale of development per day in area $i$ (decision variable, nonnegative integer)
- $v_i$ = development benefit per unit in area $i$ (from file_1_view_0.Value)
- $w_i$ = resource requirement per unit in area $i$ (from file_1_view_0.Weight)
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

---

#### Data Mapping

- Index set $I$: All file_1_view_0.ProductName
- Parameter $v_i$: file_1_view_0.Value, matched by ProductName
- Parameter $w_i$: file_1_view_0.Weight, matched by ProductName
- Parameter $C$: file_0_view_0.Capacity
- Decision variable $x_i$: scale of development per day in area $i$ (nonnegative integer), for each $i \in I$