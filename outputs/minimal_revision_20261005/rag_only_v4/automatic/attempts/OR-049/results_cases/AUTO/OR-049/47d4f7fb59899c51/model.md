ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of shelves, indexed by i. (ShelfID from file_0_view_0)
- 𝑃: Set of products, indexed by j. (ProductName from file_1_view_0)

Parameters:
- cap_i: Capacity of shelf i. (file_0_view_0, column: Capacity)
- val_j: Value per unit of product j. (file_1_view_0, column: Value)
- wt_j: Weight per unit of product j. (file_1_view_0, column: Weight)

Decision Variables:
- x_{i,j}: Number of units of product j placed on shelf i. (integer, x_{i,j} ≥ 0)

Objective:
Maximize total value across all shelves:
\[
\max \sum_{i \in S} \sum_{j \in P} val_j \cdot x_{i,j}
\]

Constraints:
1. Shelf capacity constraints (for each shelf i ∈ S):
\[
\sum_{j \in P} wt_j \cdot x_{i,j} \leq cap_i
\]

2. Nonnegativity and integrality:
\[
x_{i,j} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
\]

DATA MAPPING

- Index set 𝑆 (shelves): file_0_view_0, column ShelfID
- Index set 𝑃 (products): file_1_view_0, column ProductName
- Parameter cap_i: file_0_view_0, columns ShelfID (i), Capacity (cap_i)
- Parameter val_j: file_1_view_0, columns ProductName (j), Value (val_j)
- Parameter wt_j: file_1_view_0, columns ProductName (j), Weight (wt_j)
- Decision variable x_{i,j}: defined for all (i, j) ∈ 𝑆 × 𝑃

All data is mapped directly from the supplied tables using the specified columns.