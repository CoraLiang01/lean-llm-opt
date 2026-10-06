ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of shelves, indexed by i. (ShelfID from file_0_view_0)
- 𝑃: Set of products, indexed by j. (ProductName from file_1_view_0)

Parameters:
- cap_i: Capacity of shelf i. (Capacity from file_0_view_0, indexed by ShelfID)
- val_j: Value per unit of product j. (Value from file_1_view_0, indexed by ProductName)
- wt_j: Weight per unit of product j. (Weight from file_1_view_0, indexed by ProductName)

Decision Variables:
- x_{i,j}: Number of units of product j to place on shelf i. (integer, x_{i,j} ≥ 0, ∀i∈𝑆, j∈𝑃)

Objective:
Maximize total value across all shelves:
\[
\max \sum_{i \in S} \sum_{j \in P} val_j \cdot x_{i,j}
\]

Constraints:
1. Shelf capacity constraints (for each shelf i ∈ S):
\[
\sum_{j \in P} wt_j \cdot x_{i,j} \leq cap_i \qquad \forall i \in S
\]
2. Nonnegativity and integrality:
\[
x_{i,j} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S, j \in P
\]

DATA MAPPING

- 𝑆 (shelves): file_0_view_0.ShelfID
- cap_i: file_0_view_0.Capacity (indexed by ShelfID)
- 𝑃 (products): file_1_view_0.ProductName
- val_j: file_1_view_0.Value (indexed by ProductName)
- wt_j: file_1_view_0.Weight (indexed by ProductName)

All parameters are mapped directly from the supplied CSVs using the original business identifier columns. No data is omitted or synthesized.