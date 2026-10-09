#### Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$, with business IDs resource_id from file_0_view_0.
- $J$ = set of game genres, indexed by $j$, with business IDs item_name from file_1_view_0.
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$ (decision variable, integer, $\geq 0$).

Parameters:
- $v_j$ = item_value of genre $j$ (from file_1_view_0, column item_value).
- $a_j$ = resource_requirement of genre $j$ (from file_1_view_0, column resource_requirement).
- $c_i$ = resource_capacity of platform $i$ (from file_0_view_0, column resource_capacity).

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to:
\[
\sum_{j \in J} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$ (platforms): file_0_view_0, column resource_id
- $J$ (game genres): file_1_view_0, column item_name
- $v_j$: file_1_view_0, column item_value, key item_name
- $a_j$: file_1_view_0, column resource_requirement, key item_name
- $c_i$: file_0_view_0, column resource_capacity, key resource_id
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$ (decision variable, integer, $\geq 0$)