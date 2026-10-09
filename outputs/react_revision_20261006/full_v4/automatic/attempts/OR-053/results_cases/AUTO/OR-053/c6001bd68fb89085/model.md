##### Mathematical Model

Let:
- $I$ = set of shelves, indexed by $i$ (with ShelfID from file_0_view_0)
- $J$ = set of products, indexed by $j$ (with ProductName from file_1_view_0)

Parameters:
- $c_i$ = capacity of shelf $i$ (Capacity from file_0_view_0)
- $v_j$ = value per unit of product $j$ (Value from file_1_view_0)
- $w_j$ = weight per unit of product $j$ (Weight from file_1_view_0)

Decision variables:
- $x_{ij}$ = number of units of product $j$ to place on shelf $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j x_{ij} \leq c_i \quad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

##### Data Mapping

- $I$: ShelfID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity from file_0_view_0, indexed by ShelfID
- $v_j$: Value from file_1_view_0, indexed by ProductName
- $w_j$: Weight from file_1_view_0, indexed by ProductName
- $x_{ij}$: Number of units of product $j$ to place on shelf $i$ (decision variable, integer, $\geq 0$)