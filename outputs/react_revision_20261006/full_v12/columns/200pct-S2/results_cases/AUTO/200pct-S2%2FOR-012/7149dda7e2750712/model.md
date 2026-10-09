### Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$ (from file_0_view_0: resource_id)
- $J$ = set of game genres, indexed by $j$ (from file_1_view_0: item_name)

Parameters:
- $c_i$ = memory capacity of platform $i$ (file_0_view_0: resource_capacity, key resource_id)
- $v_j$ = value per unit of genre $j$ (file_1_view_0: item_value, key item_name)
- $a_j$ = memory requirement per unit of genre $j$ (file_1_view_0: resource_requirement, key item_name)

Decision variables:
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

---

### Data Mapping

- $I$ (platforms): file_0_view_0, column resource_id
- $c_i$: file_0_view_0, columns resource_id, resource_capacity
- $J$ (genres): file_1_view_0, column item_name
- $v_j$: file_1_view_0, columns item_name, item_value
- $a_j$: file_1_view_0, columns item_name, resource_requirement
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$ (indexed by resource_id and item_name)

All indices, parameters, and constraints are mapped directly to the supplied CSV columns and business identifiers.