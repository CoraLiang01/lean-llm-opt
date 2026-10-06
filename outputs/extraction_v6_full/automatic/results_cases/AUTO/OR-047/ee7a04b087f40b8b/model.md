ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of platforms, indexed by $i$ (PlatformId from file_0_view_0)
- $J$: set of game genres, indexed by $j$ (ProductName from file_1_view_0)

Parameters:
- $c_i$: memory capacity of platform $i$ (capacity, file_0_view_0, Capacity)
- $v_j$: value per unit of genre $j$ (value, file_1_view_0, Value)
- $w_j$: memory requirement per unit of genre $j$ (weight, file_1_view_0, Weight)

Decision Variables:
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

Data Mapping:
- $c_i$: file_0_view_0, column PlatformId = $i$, Capacity
- $v_j$: file_1_view_0, column ProductName = $j$, Value
- $w_j$: file_1_view_0, column ProductName = $j$, Weight

All indices, parameters, and constraints are mapped directly to the original data columns and business identifiers as required.