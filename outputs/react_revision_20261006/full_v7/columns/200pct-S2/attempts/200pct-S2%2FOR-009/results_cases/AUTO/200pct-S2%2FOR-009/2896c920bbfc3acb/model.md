Mathematical Model

Index Sets:
- $I$: Set of areas (from file_1_view_0.ProductName)

Parameters:
- $v_i$: Development benefit per unit in area $i$ (from file_1_view_0.Value, indexed by ProductName)
- $w_i$: Resource requirement per unit in area $i$ (from file_1_view_0.Weight, indexed by ProductName)
- $C$: Overall development capacity (from file_0_view_0.Capacity)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Scale of development per day in area $i$

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

- $I$: All file_1_view_0.ProductName
- $v_i$: file_1_view_0.Value, indexed by ProductName
- $w_i$: file_1_view_0.Weight, indexed by ProductName
- $C$: file_0_view_0.Capacity
- $x_i$: Decision variable for each $i \in I$