ABSTRACT MATHEMATICAL MODEL

Index Sets:
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

DATA MAPPING

- $I$: All PlatformId in file_0_view_0 (capacity.csv, column PlatformId)
- $J$: All ProductName in file_1_view_0 (products.csv, column ProductName)
- $c_i$: file_0_view_0, column Capacity, keyed by PlatformId
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: Integer variable for each $(i,j)$ pair

All parameters and index sets are defined directly from the returned CSV data, preserving original business identifiers and column names.