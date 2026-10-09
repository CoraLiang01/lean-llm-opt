Mathematical Model

Index Sets:
- $I$: set of platforms (resource_id from file_0_view_0)
- $J$: set of game genres (item_name from file_1_view_0)

Parameters:
- $c_i$: memory capacity of platform $i$ (resource_capacity from file_0_view_0, indexed by resource_id)
- $v_j$: value per unit of genre $j$ (item_value from file_1_view_0, indexed by item_name)
- $a_j$: memory requirement per unit of genre $j$ (resource_requirement from file_1_view_0, indexed by item_name)

Decision Variables:
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
\[
\sum_{j \in J} a_j x_{ij} \leq c_i \quad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

Data Mapping

Index Sets:
- $I$: resource_id from file_0_view_0 (capacity.csv)
- $J$: item_name from file_1_view_0 (products.csv)

Parameters:
- $c_i$: resource_capacity from file_0_view_0, indexed by resource_id
- $v_j$: item_value from file_1_view_0, indexed by item_name
- $a_j$: resource_requirement from file_1_view_0, indexed by item_name

Decision Variables:
- $x_{ij}$: integer, nonnegative, for all $i$ (resource_id), $j$ (item_name)