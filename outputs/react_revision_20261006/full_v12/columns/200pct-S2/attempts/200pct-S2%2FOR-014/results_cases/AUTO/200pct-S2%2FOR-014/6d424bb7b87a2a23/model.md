#### Mathematical Model

Let:
- $S$ = set of shelves, indexed by $i$ (from all ShelfID in file_0_view_0)
- $P$ = set of products, indexed by $j$ (from all ProductName in file_1_view_0)
- $x_{ij}$ = number of units of product $j$ placed on shelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Parameters:
- $v_j$ = value of product $j$ (Value from file_1_view_0)
- $w_j$ = weight of product $j$ (Weight from file_1_view_0)
- $C_i$ = capacity of shelf $i$ (Capacity from file_0_view_0)

Objective:
$$
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
$$

Subject to:
- Shelf capacity constraints:
$$
\sum_{j \in P} w_j \, x_{ij} \leq C_i \quad \forall i \in S
$$

- Integer nonnegativity:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
$$

---

#### Data Mapping

- $S$ (shelves): all ShelfID in file_0_view_0 (capacity.csv)
- $P$ (products): all ProductName in file_1_view_0 (products.csv)
- $C_i$: Capacity column in file_0_view_0, keyed by ShelfID
- $v_j$: Value column in file_1_view_0, keyed by ProductName
- $w_j$: Weight column in file_1_view_0, keyed by ProductName
- $x_{ij}$: decision variable for units of ProductName $j$ on ShelfID $i$