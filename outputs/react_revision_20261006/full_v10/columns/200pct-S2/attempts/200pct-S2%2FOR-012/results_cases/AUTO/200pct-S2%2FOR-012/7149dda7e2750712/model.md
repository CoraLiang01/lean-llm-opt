Mathematical Model

Index Sets:
Let $I$ be the set of platform resource IDs from file_0_view_0.resource_id.
Let $J$ be the set of game genres from file_1_view_0.item_name.

Parameters:
$c_i$ = platform memory capacity for platform $i \in I$ (file_0_view_0.resource_capacity)
$p_j$ = value per unit of genre $j \in J$ (file_1_view_0.item_value)
$a_j$ = memory requirement per unit of genre $j \in J$ (file_1_view_0.resource_requirement)

Decision Variables:
$x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of games from genre $j$ to be listed on platform $i$

Objective:
$\max \sum_{i \in I} \sum_{j \in J} p_j x_{ij}$

Subject to:
$\sum_{j \in J} a_j x_{ij} \leq c_i \quad \forall i \in I$
$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, j \in J$

Data Mapping

Index Sets:
$I$ = file_0_view_0.resource_id
$J$ = file_1_view_0.item_name

Parameters:
$c_i$ = file_0_view_0.resource_capacity for platform $i$
$p_j$ = file_1_view_0.item_value for genre $j$
$a_j$ = file_1_view_0.resource_requirement for genre $j$

Decision Variables:
$x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$ (indexed by $i \in I$, $j \in J$)