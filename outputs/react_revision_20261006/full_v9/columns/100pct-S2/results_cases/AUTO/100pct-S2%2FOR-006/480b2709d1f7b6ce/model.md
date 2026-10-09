Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from file_1_view_0, column ProductName)
- $x_i$ = number of units of vehicle type $i$ to order daily (integer, $\geq 0$)
- $v_i$ = benefit coefficient of vehicle type $i$ (from file_1_view_0, column Value)
- $w_i$ = weight (inventory space requirement) of vehicle type $i$ (from file_1_view_0, column Weight)
- $C$ = total inventory capacity (from file_0_view_0, column Capacity)

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

- $I$: All records in file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (vehicle type)