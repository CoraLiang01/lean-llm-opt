Mathematical Optimization Model

Index Sets:
- $S$: set of shelves, indexed by $i$ (from file_0_view_0, column ShelfID)
- $P$: set of products, indexed by $j$ (from file_1_view_0, column ProductName)

Parameters:
- $C_i$: capacity of shelf $i$ (from file_0_view_0, column Capacity)
- $v_j$: value per unit of product $j$ (from file_1_view_0, column Value)
- $w_j$: weight per unit of product $j$ (from file_1_view_0, column Weight)

Decision Variables:
- $x_{ij}$: number of units of product $j$ to place on shelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
\]

Subject to:
\[
\sum_{j \in P} w_j \, x_{ij} \leq C_i \qquad \forall i \in S
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
\]

Data Mapping:
- $S$ (shelves): file_0_view_0, column ShelfID
- $C_i$: file_0_view_0, column Capacity, keyed by ShelfID
- $P$ (products): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: allocation of product $j$ to shelf $i$ (decision variable, integer, nonnegative)