Mathematical Model

Index Sets:
- $I$: set of platforms, with elements $i$ corresponding to PlatformId from file_0_view_0.
- $J$: set of genres, with elements $j$ corresponding to ProductName from file_1_view_0.

Parameters:
- $c_i$: memory capacity of platform $i$ (Capacity from file_0_view_0, indexed by PlatformId).
- $v_j$: value per unit of genre $j$ (Value from file_1_view_0, indexed by ProductName).
- $w_j$: memory requirement per unit of genre $j$ (Weight from file_1_view_0, indexed by ProductName).

Decision Variables:
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

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

Index Sets:
- $I$: file_0_view_0.PlatformId
- $J$: file_1_view_0.ProductName

Parameters:
- $c_i$: file_0_view_0.Capacity, indexed by PlatformId
- $v_j$: file_1_view_0.Value, indexed by ProductName
- $w_j$: file_1_view_0.Weight, indexed by ProductName

Decision Variables:
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$ (integer, nonnegative)