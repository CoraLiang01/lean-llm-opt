Mathematical Model

Index Sets:
- $I$: set of platforms, with elements $i$ corresponding to PlatformId from file_0_view_0 (capacity.csv)
- $J$: set of game genres, with elements $j$ corresponding to ProductName from file_1_view_0 (products.csv)

Parameters:
- $c_i$: memory capacity of platform $i$ (Capacity, file_0_view_0, column "Capacity", key "PlatformId")
- $v_j$: value per unit of genre $j$ (Value, file_1_view_0, column "Value", key "ProductName")
- $w_j$: memory requirement per unit of genre $j$ (Weight, file_1_view_0, column "Weight", key "ProductName")

Decision Variables:
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to:

Platform memory capacity constraints:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$: PlatformId from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: file_0_view_0, column "Capacity", key "PlatformId"
- $v_j$: file_1_view_0, column "Value", key "ProductName"
- $w_j$: file_1_view_0, column "Weight", key "ProductName"
- $x_{ij}$: number of units of games from genre $j$ to be listed on platform $i$ (decision variable, indexed by PlatformId and ProductName)