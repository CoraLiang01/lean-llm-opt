ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of areas (from file_1_view_0, column ProductName)

Parameters:
- $v_i$: Development benefit per unit in area $i$ (file_1_view_0, column Value, key ProductName)
- $w_i$: Development capacity required per unit in area $i$ (file_1_view_0, column Weight, key ProductName)
- $C$: Overall development capacity (file_0_view_0, column Capacity)

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

DATA MAPPING

- $I$: file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: scale of development per day in area $i$ (decision variable, indexed by ProductName)