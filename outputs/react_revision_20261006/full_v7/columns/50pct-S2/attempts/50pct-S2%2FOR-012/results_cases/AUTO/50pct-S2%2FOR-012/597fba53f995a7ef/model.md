#### Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$, with platform IDs from column resource_id in file_0_view_0.
- $J$ = set of game genres, indexed by $j$, with genre names from column item_name in file_1_view_0.

Parameters:
- $c_i$ = memory capacity of platform $i$ (resource_capacity from file_0_view_0, keyed by resource_id).
- $v_j$ = value per unit of genre $j$ (item_value from file_1_view_0, keyed by item_name).
- $a_j$ = memory requirement per unit of genre $j$ (resource_requirement from file_1_view_0, keyed by item_name).

Decision Variables:
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

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

#### Data Mapping

- $I$ (platforms): resource_id in file_0_view_0 (capacity.csv)
- $c_i$: resource_capacity in file_0_view_0, keyed by resource_id
- $J$ (genres): item_name in file_1_view_0 (products.csv)
- $v_j$: item_value in file_1_view_0, keyed by item_name
- $a_j$: resource_requirement in file_1_view_0, keyed by item_name
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$ (decision variable, integer, nonnegative)