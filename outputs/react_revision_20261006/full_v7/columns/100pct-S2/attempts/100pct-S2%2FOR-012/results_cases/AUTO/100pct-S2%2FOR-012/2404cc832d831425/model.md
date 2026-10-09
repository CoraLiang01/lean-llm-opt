#### Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$ (from all resource_id in file_0_view_0)
- $J$ = set of game genres, indexed by $j$ (from all item_name in file_1_view_0)
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$ (decision variable, integer, $\geq 0$)
- $v_j$ = value of one unit of genre $j$ (item_value from file_1_view_0)
- $a_j$ = memory requirement of one unit of genre $j$ (resource_requirement from file_1_view_0)
- $c_i$ = memory capacity of platform $i$ (resource_capacity from file_0_view_0)

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Subject to:**
\[
\sum_{j \in J} a_j \, x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$ (platforms): resource_id from file_0_view_0 (capacity.csv)
- $c_i$: resource_capacity from file_0_view_0, indexed by resource_id
- $J$ (genres): item_name from file_1_view_0 (products.csv)
- $v_j$: item_value from file_1_view_0, indexed by item_name
- $a_j$: resource_requirement from file_1_view_0, indexed by item_name
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$ (decision variable, integer, $\geq 0$)