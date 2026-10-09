##### Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$ (from PlatformID in file_0_view_0)
- $J$ = set of games/genres, indexed by $j$ (from ProductName in file_1_view_0)

Parameters:
- $c_i$ = capacity of platform $i$ (Capacity from file_0_view_0, indexed by PlatformID)
- $v_j$ = value per unit of game $j$ (Value from file_1_view_0, indexed by ProductName)
- $w_j$ = memory requirement per unit of game $j$ (Weight from file_1_view_0, indexed by ProductName)

Decision variables:
- $x_{ij}$ = number of units of game $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

##### Data Mapping

- $I$ (platforms): PlatformID from file_0_view_0 (capacity.csv)
- $J$ (games/genres): ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity from file_0_view_0, indexed by PlatformID
- $v_j$: Value from file_1_view_0, indexed by ProductName
- $w_j$: Weight from file_1_view_0, indexed by ProductName
- $x_{ij}$: number of units of game $j$ to be listed on platform $i$ (decision variable, integer, $\geq 0$)