Mathematical Model

Sets:
- $I$: set of drug types, indexed by $i$ (from all ProductName in file_1_view_0)

Parameters:
- $v_i$: benefit coefficient of drug $i$ (Value, file_1_view_0)
- $w_i$: weight per unit of drug $i$ (Weight, file_1_view_0)
- $C$: overall inventory capacity (Capacity, file_0_view_0)

Decision Variables:
- $x_i$: number of units of drug $i$ to order daily ($x_i \in \mathbb{Z}_{\geq 0}$)

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
- $v_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: decision variable for each $i \in I$