Mathematical Model

Index Sets:
- $I$: set of platforms, indexed by $i$ (from file_0_view_0.resource_id)
- $J$: set of game genres, indexed by $j$ (from file_1_view_0.item_name)

Parameters:
- $c_i$: memory capacity of platform $i$ (file_0_view_0.resource_capacity, for each $i \in I$)
- $v_j$: value per unit of genre $j$ (file_1_view_0.item_value, for each $j \in J$)
- $a_j$: memory requirement per unit of genre $j$ (file_1_view_0.resource_requirement, for each $j \in J$)

Decision Variables:
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

- $I$: file_0_view_0.resource_id
- $J$: file_1_view_0.item_name
- $c_i$: file_0_view_0.resource_capacity, matched by $i$
- $v_j$: file_1_view_0.item_value, matched by $j$
- $a_j$: file_1_view_0.resource_requirement, matched by $j$
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$ (decision variable, integer, nonnegative)