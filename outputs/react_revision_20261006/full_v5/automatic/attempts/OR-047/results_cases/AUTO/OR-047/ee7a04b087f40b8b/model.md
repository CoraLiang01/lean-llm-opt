Mathematical Model

Sets:
- $I$: Set of platforms, indexed by $i$ (from PlatformId in file_0_view_0)
- $J$: Set of game genres, indexed by $j$ (from ProductName in file_1_view_0)

Parameters:
- $c_i$: Memory capacity of platform $i$ (Capacity from file_0_view_0)
- $v_j$: Value per unit of genre $j$ (Value from file_1_view_0)
- $w_j$: Memory requirement per unit of genre $j$ (Weight from file_1_view_0)

Decision Variables:
- $x_{ij}$: Number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

Data Mapping

- $I$: PlatformId from file_0_view_0
- $J$: ProductName from file_1_view_0
- $c_i$: Capacity column in file_0_view_0, matched by PlatformId $i$
- $v_j$: Value column in file_1_view_0, matched by ProductName $j$
- $w_j$: Weight column in file_1_view_0, matched by ProductName $j$
- $x_{ij}$: Number of units of games from genre $j$ to be listed on platform $i$ (decision variable)