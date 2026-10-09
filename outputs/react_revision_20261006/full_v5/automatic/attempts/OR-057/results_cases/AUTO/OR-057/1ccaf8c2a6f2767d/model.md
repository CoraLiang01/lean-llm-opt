Mathematical Optimization Model

Index Sets:
- $I$: Set of platforms, indexed by $i$ (PlatformID from file_0_view_0)
- $J$: Set of games, indexed by $j$ (ProductName from file_1_view_0)

Parameters:
- $c_i$: Capacity of platform $i$ (Capacity from file_0_view_0)
- $v_j$: Value of game $j$ (Value from file_1_view_0)
- $w_j$: Memory requirement (Weight) of game $j$ (Weight from file_1_view_0)

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
- $I$: PlatformID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: file_0_view_0, column Capacity, keyed by PlatformID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: Number of units of game $j$ to be listed on platform $i$ (decision variable, integer, $\geq 0$)