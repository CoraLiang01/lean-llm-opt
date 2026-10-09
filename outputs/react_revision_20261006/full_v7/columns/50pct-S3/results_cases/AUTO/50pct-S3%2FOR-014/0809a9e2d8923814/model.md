#### Mathematical Model

Let:
- $I$ = set of shelves, indexed by $i$ (from all ShelfID in file_0_view_0)
- $J$ = set of products, indexed by $j$ (from all ProductName in file_1_view_0)

Parameters:
- $c_i$ = capacity of shelf $i$ (file_0_view_0, column Capacity, key ShelfID)
- $v_j$ = value per unit of product $j$ (file_1_view_0, column Value, key ProductName)
- $w_j$ = weight per unit of product $j$ (file_1_view_0, column Weight, key ProductName)

Decision variables:
- $x_{ij}$ = number of units of product $j$ to place on shelf $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$: All ShelfID from file_0_view_0 (capacity.csv), column ShelfID
- $J$: All ProductName from file_1_view_0 (products.csv), column ProductName
- $c_i$: file_0_view_0 (capacity.csv), column Capacity, key ShelfID
- $v_j$: file_1_view_0 (products.csv), column Value, key ProductName
- $w_j$: file_1_view_0 (products.csv), column Weight, key ProductName
- $x_{ij}$: Number of units of product $j$ to place on shelf $i$ (decision variable, nonnegative integer)