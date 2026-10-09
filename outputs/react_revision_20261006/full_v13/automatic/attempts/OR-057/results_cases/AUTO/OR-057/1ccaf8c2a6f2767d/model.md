##### Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$ (from PlatformID in file_0_view_0)
- $J$ = set of games, indexed by $j$ (from ProductName in file_1_view_0)

Parameters:
- $c_i$ = capacity of platform $i$ (Capacity from file_0_view_0, key PlatformID)
- $v_j$ = value per unit of game $j$ (Value from file_1_view_0, key ProductName)
- $w_j$ = memory requirement per unit of game $j$ (Weight from file_1_view_0, key ProductName)

Decision variables:
- $x_{ij}$ = number of units of game $j$ to list on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

- $I$: All PlatformID in file_0_view_0 (capacity.csv), column PlatformID
- $J$: All ProductName in file_1_view_0 (products.csv), column ProductName
- $c_i$: file_0_view_0, column Capacity, key PlatformID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $x_{ij}$: Number of units of game $j$ to list on platform $i$ (decision variable, integer, $\geq 0$)