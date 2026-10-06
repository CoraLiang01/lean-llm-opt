ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: Set of platforms, indexed by $i$ (from PlatformID in file_0_view_0)
- $J$: Set of games, indexed by $j$ (from ProductName in file_1_view_0)

Parameters:
- $c_i$: Memory capacity of platform $i$ (capacity, file_0_view_0, Capacity)
- $v_j$: Value of game $j$ (value, file_1_view_0, Value)
- $w_j$: Memory requirement of game $j$ (weight, file_1_view_0, Weight)

Decision Variables:
- $x_{ij}$: Number of units of game $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

- $c_i$: file_0_view_0, column PlatformID = $i$, Capacity
- $v_j$: file_1_view_0, column ProductName = $j$, Value
- $w_j$: file_1_view_0, column ProductName = $j$, Weight

Each $x_{ij}$ is the integer number of units of game $j$ to list on platform $i$. All platforms and games from the source files are included, with their original identifiers and coefficients.