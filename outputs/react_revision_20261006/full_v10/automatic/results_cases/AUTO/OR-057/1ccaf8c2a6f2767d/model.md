Mathematical Model

Index Sets:
Let $I$ be the set of PlatformIDs from file_0_view_0[PlatformID].
Let $J$ be the set of ProductNames from file_1_view_0[ProductName].

Parameters:
For each $i \in I$, let $C_i =$ file_0_view_0[Capacity] for PlatformID $i$.
For each $j \in J$, let $v_j =$ file_1_view_0[Value] for ProductName $j$.
For each $j \in J$, let $w_j =$ file_1_view_0[Weight] for ProductName $j$.

Decision Variables:
For each $i \in I$, $j \in J$:
$x_{ij} \in \mathbb{Z}_{\geq 0}$ = number of units of game $j$ to be listed on platform $i$.

Objective:
$\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}$

Subject to:
For all $i \in I$:
$\sum_{j \in J} w_j x_{ij} \leq C_i$

For all $i \in I$, $j \in J$:
$x_{ij} \in \mathbb{Z}_{\geq 0}$

Data Mapping

Index Sets:
$I$ = file_0_view_0[PlatformID]
$J$ = file_1_view_0[ProductName]

Parameters:
$C_i$ = file_0_view_0[Capacity] where PlatformID $= i$
$v_j$ = file_1_view_0[Value] where ProductName $= j$
$w_j$ = file_1_view_0[Weight] where ProductName $= j$

Decision Variables:
$x_{ij}$: number of units of game $j$ to be listed on platform $i$

All data is mapped directly from the returned CSV columns as specified.