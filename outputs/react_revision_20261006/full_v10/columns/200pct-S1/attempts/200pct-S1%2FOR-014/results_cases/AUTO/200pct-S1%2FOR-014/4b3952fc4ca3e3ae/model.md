ABSTRACT MATHEMATICAL MODEL

Index Sets:
- Let $S$ be the set of shelves, with elements indexed by $i$ and identified by ShelfID from file_0_view_0.
- Let $P$ be the set of products, with elements indexed by $j$ and identified by ProductName from file_1_view_0.

Parameters:
- $c_i$: Capacity of shelf $i$ (from file_0_view_0, column Capacity).
- $v_j$: Value per unit of product $j$ (from file_1_view_0, column Value).
- $w_j$: Weight per unit of product $j$ (from file_1_view_0, column Weight).

Decision Variables:
- $x_{ij}$: Number of units of product $j$ to place on shelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

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

DATA MAPPING

Index Sets:
- $S$: file_0_view_0, column ShelfID
- $P$: file_1_view_0, column ProductName

Parameters:
- $c_i$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName

Decision Variables:
- $x_{ij}$: Number of units of product $j$ to place on shelf $i$ (indexed by ShelfID and ProductName)

Objective:
- Maximize total value of products allocated to all shelves.

Constraints:
- For each shelf $i$, the total weight of products allocated does not exceed its capacity $c_i$.
- All $x_{ij}$ are nonnegative integers.