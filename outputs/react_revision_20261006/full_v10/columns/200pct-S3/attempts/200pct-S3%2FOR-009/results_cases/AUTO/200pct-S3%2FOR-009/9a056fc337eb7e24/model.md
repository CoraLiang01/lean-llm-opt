Mathematical Model

Index Sets:
- Let $I$ be the set of areas, with each area identified by ProductName from file_1_view_0.

Parameters:
- $v_i$: Value (development benefit) of area $i$ (from Value in file_1_view_0)
- $w_i$: Weight (resource requirement) of area $i$ (from Weight in file_1_view_0)
- $C$: Overall development capacity (from Capacity in file_0_view_0)

Decision Variables:
- $x_i \geq 0$, integer: scale of development per day in area $i$

Objective:
$\max \sum_{i \in I} v_i x_i$

Subject to:
$\sum_{i \in I} w_i x_i \leq C$

$x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$

Data Mapping

Index Sets:
- $I$: All ProductName values in file_1_view_0

Parameters:
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity

Variables:
- $x_i$: scale of development per day in area $i$ (nonnegative integer), indexed by ProductName in file_1_view_0

Objective:
- Maximize total development benefit: $\sum_{i \in I} v_i x_i$

Constraints:
- Overall development capacity: $\sum_{i \in I} w_i x_i \leq C$
- Nonnegativity and integrality: $x_i \in \mathbb{Z}_{\geq 0}$ for all $i \in I$