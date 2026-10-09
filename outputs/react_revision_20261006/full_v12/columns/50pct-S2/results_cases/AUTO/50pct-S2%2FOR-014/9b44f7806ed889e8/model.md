### Mathematical Model

Let:
- $I$ = set of products (indexed by $i$), from ProductName in file_1_view_0
- $J$ = set of shelves (indexed by $j$), from ShelfID in file_0_view_0

Parameters:
- $v_i$ = Value of product $i$ (file_1_view_0, column Value)
- $w_i$ = Weight of product $i$ (file_1_view_0, column Weight)
- $C_j$ = Capacity of shelf $j$ (file_0_view_0, column Capacity)

Decision variables:
- $x_{ij}$ = number of units of product $i$ placed on shelf $j$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

### Data Mapping

- $I$: All ProductName in file_1_view_0
- $J$: All ShelfID in file_0_view_0
- $v_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C_j$: file_0_view_0, column Capacity, key ShelfID
- $x_{ij}$: number of units of product $i$ on shelf $j$, integer, $\geq 0$