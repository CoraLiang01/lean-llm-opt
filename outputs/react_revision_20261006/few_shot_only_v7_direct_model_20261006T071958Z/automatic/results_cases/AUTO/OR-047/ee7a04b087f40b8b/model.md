ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of platforms (indexed by $i$), from file_0_view_0.PlatformId
- $J$: set of game genres (indexed by $j$), from file_1_view_0.ProductName

Parameters:
- $c_i$: memory capacity of platform $i$, from file_0_view_0.Capacity
- $v_j$: value per unit of genre $j$, from file_1_view_0.Value
- $w_j$: memory requirement per unit of genre $j$, from file_1_view_0.Weight

Decision Variables:
- $x_{ij}$: number of units of genre $j$ to list on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

DATA MAPPING

- $I$: file_0_view_0.PlatformId
- $J$: file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity, keyed by PlatformId
- $v_j$: file_1_view_0.Value, keyed by ProductName
- $w_j$: file_1_view_0.Weight, keyed by ProductName
- $x_{ij}$: number of units of genre $j$ to list on platform $i$ (decision variable)