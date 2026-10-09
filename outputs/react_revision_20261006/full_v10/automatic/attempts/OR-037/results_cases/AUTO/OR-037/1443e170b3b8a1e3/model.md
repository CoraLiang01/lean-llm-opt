Mathematical Model

Index Sets:
Let $I$ be the set of vehicle types, with each $i \in I$ corresponding to a unique ProductName from file_1_view_0.

Parameters:
For each $i \in I$:
- $p_i$: profit per unit of vehicle $i$ (Value from file_1_view_0, column Value)
- $w_i$: inventory weight per unit of vehicle $i$ (Weight from file_1_view_0, column Weight)

Let $C$ be the overall inventory capacity (Capacity from file_0_view_0, column Capacity).

Decision Variables:
For each $i \in I$:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of vehicles of type $i$ to order per day

Objective:
$\max \sum_{i \in I} p_i x_i$

Subject to:
$\sum_{i \in I} w_i x_i \leq C$

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

Data Mapping

Index Sets:
- $I$: All ProductName values in file_1_view_0

Parameters:
- $p_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity

Decision Variables:
- $x_i$: number of vehicles of type $i$ to order per day, for each $i \in I$

Objective:
- $\sum_{i \in I} p_i x_i$ (maximize total profit)

Constraint:
- $\sum_{i \in I} w_i x_i \leq C$ (total inventory weight does not exceed overall capacity)

Variable Domains:
- $x_i \in \mathbb{Z}_{\geq 0}$ for all $i \in I$