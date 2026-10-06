ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: Set of platforms, indexed by $i$ (from PlatformId in file_0_view_0)
- $J$: Set of game genres, indexed by $j$ (from ProductName in file_1_view_0)

Parameters:
- $c_i$: Memory capacity of platform $i$ (Capacity from file_0_view_0, indexed by PlatformId)
- $v_j$: Value per unit of genre $j$ (Value from file_1_view_0, indexed by ProductName)
- $w_j$: Memory requirement per unit of genre $j$ (Weight from file_1_view_0, indexed by ProductName)

Decision Variables:
- $x_{ij}$: Number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

---

DATA MAPPING

- $I$ (platforms): file_0_view_0, column PlatformId
- $c_i$: file_0_view_0, column Capacity, indexed by PlatformId
- $J$ (genres): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, column Value, indexed by ProductName
- $w_j$: file_1_view_0, column Weight, indexed by ProductName
- $x_{ij}$: Number of units of games from genre $j$ on platform $i$ (decision variable, integer, nonnegative)