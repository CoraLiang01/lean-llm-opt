##### Mathematical Model

Let:
- $I$ = set of platforms, indexed by $i$ (from all PlatformId in file_0_view_0)
- $J$ = set of game genres, indexed by $j$ (from all ProductName in file_1_view_0)
- $v_j$ = value of genre $j$ (from Value in file_1_view_0)
- $w_j$ = memory requirement of genre $j$ (from Weight in file_1_view_0)
- $C_i$ = memory capacity of platform $i$ (from Capacity in file_0_view_0)
- $x_{ij}$ = number of units of games from genre $j$ to be listed on platform $i$ (decision variable, integer, $\geq 0$)

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

##### Data Mapping

- $I$: PlatformId from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $v_j$: Value from file_1_view_0, column Value, key ProductName
- $w_j$: Weight from file_1_view_0, column Weight, key ProductName
- $C_i$: Capacity from file_0_view_0, column Capacity, key PlatformId
- $x_{ij}$: integer, $\geq 0$, for all $(i,j) \in I \times J$