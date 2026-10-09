Mathematical Model

Sets:
- $I$: set of bread types, indexed by $i$ (from all ProductName in file_1_view_0)

Parameters:
- $v_i$: expected profit per unit of bread $i$ (Value from file_1_view_0, column Value)
- $w_i$: storage weight per unit of bread $i$ (Weight from file_1_view_0, column Weight)
- $C$: total storage capacity (Capacity from file_0_view_0, column Capacity)

Decision Variables:
- $x_i$: number of units of bread $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

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
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: decision variable for each $i \in I$