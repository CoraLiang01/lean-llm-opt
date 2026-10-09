#### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from ProductName in file_1_view_0)
- $J$ = set of shelves, indexed by $j$ (from ShelfID in file_0_view_0)

Parameters:
- $v_i$ = value of one unit of product $i$ (Value from file_1_view_0)
- $w_i$ = weight of one unit of product $i$ (Weight from file_1_view_0)
- $C_j$ = capacity of shelf $j$ (Capacity from file_0_view_0)

Decision variables:
- $x_{ij}$ = number of units of product $i$ to place on shelf $j$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

Objective:
\[
\max \sum_{j \in J} \sum_{i \in I} v_i \, x_{ij}
\]

Subject to:
\[
\sum_{i \in I} w_i \, x_{ij} \leq C_j \qquad \forall j \in J
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$ (products): file_1_view_0.ProductName
- $J$ (shelves): file_0_view_0.ShelfID
- $v_i$: file_1_view_0.Value, keyed by ProductName
- $w_i$: file_1_view_0.Weight, keyed by ProductName
- $C_j$: file_0_view_0.Capacity, keyed by ShelfID
- $x_{ij}$: number of units of product $i$ on shelf $j$ (decision variable, integer, nonnegative)