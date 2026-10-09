Mathematical Model

Index Sets:
Let $I$ be the set of products, with each $i \in I$ corresponding to a ProductName from file_1_view_0.

Parameters:
For each $i \in I$:
- $v_i$: Value of product $i$ (file_1_view_0, column Value)
- $w_i$: Weight of product $i$ (file_1_view_0, column Weight)

Let $C$ be the overall stock capacity (file_0_view_0, column Capacity).

Decision Variables:
For each $i \in I$:
- $x_i$: number of units of product $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
$\max \sum_{i \in I} v_i x_i$

Subject to:
$\sum_{i \in I} w_i x_i \leq C$

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

Data Mapping

Index Sets:
- $I$: All ProductName in file_1_view_0

Parameters:
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity

Decision Variables:
- $x_i$: number of units of product $i$ to order each day ($i$ indexed by ProductName from file_1_view_0)

Objective:
- Maximize total value: $\sum_{i \in I} v_i x_i$

Constraint:
- Total weight ordered does not exceed overall stock capacity: $\sum_{i \in I} w_i x_i \leq C$