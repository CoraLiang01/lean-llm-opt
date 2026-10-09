Mathematical Model

Sets:
- $I$: set of platforms, indexed by $i$ (from all resource_id in file_0_view_0)
- $J$: set of game genres, indexed by $j$ (from all item_name in file_1_view_0)

Parameters:
- $c_i$: memory capacity of platform $i$ (resource_capacity from file_0_view_0)
- $v_j$: value per unit of genre $j$ (item_value from file_1_view_0)
- $a_j$: memory requirement per unit of genre $j$ (resource_requirement from file_1_view_0)

Decision Variables:
- $x_{ij}$: number of units of genre $j$ to list on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

- $I$: All resource_id in file_0_view_0 (capacity.csv)
- $J$: All item_name in file_1_view_0 (products.csv)
- $c_i$: resource_capacity in file_0_view_0, matched by resource_id
- $v_j$: item_value in file_1_view_0, matched by item_name
- $a_j$: resource_requirement in file_1_view_0, matched by item_name
- $x_{ij}$: number of units of genre $j$ to list on platform $i$ (decision variable, integer, $\geq 0$)