Mathematical Model

Index Sets:
- $S$: set of shelves, indexed by $i$ (from all ShelfID in file_0_view_0)
- $P$: set of products, indexed by $j$ (from all ProductName in file_1_view_0)

Parameters:
- $c_i$: capacity of shelf $i$ (Capacity from file_0_view_0, key: ShelfID)
- $v_j$: value per unit of product $j$ (Value from file_1_view_0, key: ProductName)
- $w_j$: weight per unit of product $j$ (Weight from file_1_view_0, key: ProductName)

Decision Variables:
- $x_{ij}$: number of units of product $j$ placed on shelf $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
\]

Subject to:
\[
\sum_{j \in P} w_j \, x_{ij} \leq c_i \qquad \forall i \in S
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
\]

Data Mapping

- $S$ (shelves): all ShelfID in file_0_view_0 (capacity.csv)
- $P$ (products): all ProductName in file_1_view_0 (products.csv)
- $c_i$: file_0_view_0, column Capacity, key ShelfID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $x_{ij}$: number of units of product $j$ on shelf $i$ (decision variable, nonnegative integer)