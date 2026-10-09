#### Mathematical Model

Let $I$ be the set of areas, indexed by $i$ (from file_1_view_0, column ProductName).

Parameters:
- $v_i$: development benefit of area $i$ (file_1_view_0, column Value)
- $w_i$: resource requirement per unit development in area $i$ (file_1_view_0, column Weight)
- $C$: overall development capacity (file_0_view_0, column Capacity)

Decision variables:
- $x_i \geq 0$, integer: scale of development per day in area $i$

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

- $I$: file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: scale of development per day in area $i$ (decision variable, indexed by ProductName)