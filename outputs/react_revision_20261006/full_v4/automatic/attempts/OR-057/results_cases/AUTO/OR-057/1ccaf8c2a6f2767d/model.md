Mathematical Model

Sets:
- $I$: set of platforms, indexed by $i$ (PlatformID from file_0_view_0)
- $J$: set of games, indexed by $j$ (ProductName from file_1_view_0)

Parameters:
- $c_i$: memory capacity of platform $i$ (Capacity from file_0_view_0)
- $v_j$: value of game $j$ (Value from file_1_view_0)
- $w_j$: memory requirement of game $j$ (Weight from file_1_view_0)

Decision Variables:
- $x_{ij}$: number of units of game $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$: PlatformID from file_0_view_0
- $J$: ProductName from file_1_view_0
- $c_i$: Capacity from file_0_view_0, mapped by PlatformID
- $v_j$: Value from file_1_view_0, mapped by ProductName
- $w_j$: Weight from file_1_view_0, mapped by ProductName
- $x_{ij}$: integer variable for each $(i,j)$ pair

All sets, parameters, and variables are defined over the full set of records returned from the respective files.