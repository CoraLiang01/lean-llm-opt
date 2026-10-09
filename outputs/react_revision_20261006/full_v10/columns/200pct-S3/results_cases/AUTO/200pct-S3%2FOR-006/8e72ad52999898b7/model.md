Mathematical Model

Index Sets:
Let $I$ be the set of vehicle types, with each $i \in I$ corresponding to a unique ProductName from file_1_view_0.

Parameters:
For each $i \in I$:
- $v_i$: Value of vehicle type $i$ (file_1_view_0, column Value)
- $w_i$: Weight (inventory space required) of vehicle type $i$ (file_1_view_0, column Weight)

Let $C$ be the total inventory capacity (file_0_view_0, column Capacity).

Decision Variables:
For each $i \in I$:
- $x_i$: number of units of vehicle type $i$ to order daily ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
$\max \sum_{i \in I} v_i x_i$

Subject to:
$\sum_{i \in I} w_i x_i \leq C$

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

Data Mapping

Index Sets:
- $I$: All ProductName values in file_1_view_0

Parameters:
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity

Decision Variables:
- $x_i$: integer, for each $i \in I$

Objective:
- $\sum_{i \in I} v_i x_i$

Constraint:
- $\sum_{i \in I} w_i x_i \leq C$

Variable domain:
- $x_i \in \mathbb{Z}_{\geq 0}$ for all $i \in I$