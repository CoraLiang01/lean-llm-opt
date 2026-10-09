Mathematical Model

Sets:
- $I$: set of products, indexed by $i$ (from ProductName in file_1_view_0)

Parameters:
- $v_i$: value (benefit) per unit of product $i$ (Value column, file_1_view_0)
- $w_i$: weight (stock space required) per unit of product $i$ (Weight column, file_1_view_0)
- $C$: overall stock capacity (Capacity column, file_0_view_0)

Decision Variables:
- $x_i$: number of units of product $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

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

- $I$: All ProductName in file_1_view_0
- $v_i$: Value column in file_1_view_0, mapped by ProductName
- $w_i$: Weight column in file_1_view_0, mapped by ProductName
- $C$: Capacity column in file_0_view_0
- $x_i$: Decision variable for each $i \in I$ (product in file_1_view_0)