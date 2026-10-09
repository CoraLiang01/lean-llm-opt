##### Mathematical Model

Let:
- $S$ = set of shelves, indexed by $i$ (with ShelfID from file_0_view_0)
- $P$ = set of products, indexed by $j$ (with ProductName from file_1_view_0)

Parameters:
- $c_i$ = capacity of shelf $i$ (Capacity from file_0_view_0)
- $v_j$ = value per unit of product $j$ (Value from file_1_view_0)
- $w_j$ = weight per unit of product $j$ (Weight from file_1_view_0)

Decision variables:
- $x_{ij}$ = number of units of product $j$ placed on shelf $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

---

##### Data Mapping

- $S$ (shelves): file_0_view_0, column ShelfID
- $P$ (products): file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: decision variable for allocation of product $j$ to shelf $i$ (indexed by ShelfID and ProductName)