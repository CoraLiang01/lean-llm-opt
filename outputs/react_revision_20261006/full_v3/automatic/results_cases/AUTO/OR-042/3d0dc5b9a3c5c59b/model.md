Mathematical Model

Sets:
- $I$: set of drug types, indexed by $i$ (from file_1_view_0, column ProductName)

Parameters:
- $v_i$: benefit coefficient of drug type $i$ (file_1_view_0, column Value)
- $w_i$: weight per unit of drug type $i$ (file_1_view_0, column Weight)
- $C$: overall inventory capacity (file_0_view_0, column Capacity)

Decision Variables:
- $x_i$: number of units of drug type $i$ to order daily ($x_i \in \mathbb{Z}_{\geq 0}$)

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

- $I$: file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: decision variable for each $i \in I$