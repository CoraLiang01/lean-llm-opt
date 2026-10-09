Mathematical Model

Index Sets:
- $I$: set of platforms, with elements $i$ corresponding to resource_id from file_0_view_0.
- $J$: set of game genres, with elements $j$ corresponding to item_name from file_1_view_0.

Parameters:
- $c_i$: memory capacity of platform $i$ (resource_capacity from file_0_view_0, indexed by resource_id).
- $v_j$: value per unit of genre $j$ (item_value from file_1_view_0, indexed by item_name).
- $a_j$: memory requirement per unit of genre $j$ (resource_requirement from file_1_view_0, indexed by item_name).

Decision Variables:
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

Subject to:
\[
\sum_{j \in J} a_j \, x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

Data Mapping

Index Sets:
- $I$: All resource_id in file_0_view_0 (capacity.csv)
- $J$: All item_name in file_1_view_0 (products.csv)

Parameters:
- $c_i$: file_0_view_0.resource_capacity, indexed by file_0_view_0.resource_id
- $v_j$: file_1_view_0.item_value, indexed by file_1_view_0.item_name
- $a_j$: file_1_view_0.resource_requirement, indexed by file_1_view_0.item_name

Decision Variables:
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$ (integer, nonnegative)