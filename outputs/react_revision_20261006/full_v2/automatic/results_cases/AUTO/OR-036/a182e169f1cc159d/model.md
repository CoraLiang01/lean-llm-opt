Mathematical Model

Sets:
- $I$: set of vehicle types, indexed by $i$ (from all ProductName in file_1_view_0)

Parameters:
- $v_i$: benefit coefficient of vehicle type $i$ (Value from file_1_view_0, column Value, key ProductName)
- $w_i$: inventory weight/unit for vehicle type $i$ (Weight from file_1_view_0, column Weight, key ProductName)
- $C$: total inventory capacity (Capacity from file_0_view_0, column Capacity)

Decision Variables:
- $x_i$: number of units of vehicle type $i$ to order daily ($x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$)

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