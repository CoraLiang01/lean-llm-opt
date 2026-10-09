##### Mathematical Model

Let:
- $I$ = set of shelves, indexed by $i$, with ShelfID from file_0_view_0.
- $J$ = set of products, indexed by $j$, with ProductName from file_1_view_0.

Parameters:
- $c_i$ = capacity of shelf $i$ (from file_0_view_0, column Capacity).
- $v_j$ = value per unit of product $j$ (from file_1_view_0, column Value).
- $w_j$ = weight per unit of product $j$ (from file_1_view_0, column Weight).

Decision variables:
- $x_{ij}$ = number of units of product $j$ to place on shelf $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$.

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

---

##### Data Mapping

- $I$ (shelves): file_0_view_0, column ShelfID
- $J$ (products): file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: number of units of product $j$ on shelf $i$ (decision variable, integer, nonnegative)