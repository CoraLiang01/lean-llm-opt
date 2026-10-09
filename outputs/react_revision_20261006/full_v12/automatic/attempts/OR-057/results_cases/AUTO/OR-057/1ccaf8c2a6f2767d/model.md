#### Mathematical Model

Let:
- $I$ = set of platforms (indexed by $i$), from file_0_view_0[PlatformID]
- $J$ = set of games (indexed by $j$), from file_1_view_0[ProductName]
- $c_i$ = capacity of platform $i$, from file_0_view_0[Capacity]
- $v_j$ = value of game $j$, from file_1_view_0[Value]
- $w_j$ = memory requirement of game $j$, from file_1_view_0[Weight]
- $x_{ij}$ = number of units of game $j$ to be listed on platform $i$ (decision variable, integer, $\geq 0$)

Objective:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

Subject to:
$$
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

#### Data Mapping

- $I$: file_0_view_0[PlatformID]
- $J$: file_1_view_0[ProductName]
- $c_i$: file_0_view_0[Capacity], keyed by PlatformID
- $v_j$: file_1_view_0[Value], keyed by ProductName
- $w_j$: file_1_view_0[Weight], keyed by ProductName
- $x_{ij}$: number of units of game $j$ to be listed on platform $i$ (decision variable, integer, $\geq 0$)