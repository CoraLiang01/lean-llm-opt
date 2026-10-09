#### Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$, with business IDs PlatformId from file_0_view_0.
- $J$ = set of game genres, indexed by $j$, with business IDs ProductName from file_1_view_0.

Parameters:
- $c_i$ = Capacity of platform $i$ (from file_0_view_0, column Capacity).
- $v_j$ = Value per unit of genre $j$ (from file_1_view_0, column Value).
- $w_j$ = Memory requirement per unit of genre $j$ (from file_1_view_0, column Weight).

Decision variables:
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

Objective:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

Subject to:
- Platform memory capacity constraints:
$$
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
$$

- Integer and nonnegativity constraints:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

---

#### Data Mapping

- $I$ (platforms): file_0_view_0, column PlatformId
- $J$ (genres): file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by PlatformId
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: number of units of genre $j$ on platform $i$ (decision variable, indexed by PlatformId and ProductName)