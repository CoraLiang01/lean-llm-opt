Mathematical Model (Abstract Formulation)

Index Sets:
- 𝑆: Set of shelves (indexed by i), from file_0_view_0.ShelfID
- 𝑃: Set of products (indexed by j), from file_1_view_0.ProductName

Parameters:
- cap_i: Capacity of shelf i ∈ 𝑆 (file_0_view_0.Capacity)
- val_j: Value per unit of product j ∈ 𝑃 (file_1_view_0.Value)
- wt_j: Weight per unit of product j ∈ 𝑃 (file_1_view_0.Weight)

Decision Variables:
- x_{i,j}: Number of units of product j allocated to shelf i (integer, x_{i,j} ≥ 0)

Objective:
Maximize total value across all shelves:
\[
\max \sum_{i \in S} \sum_{j \in P} val_j \cdot x_{i,j}
\]

Subject to:
1. Shelf capacity constraints (for each shelf i ∈ 𝑆):
\[
\sum_{j \in P} wt_j \cdot x_{i,j} \leq cap_i \quad \forall i \in S
\]
2. Nonnegativity and integrality:
\[
x_{i,j} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
\]

Data Mapping

Index Sets:
- 𝑆: All ShelfID values in file_0_view_0.ShelfID (capacity.csv, preserve row order)
- 𝑃: All ProductName values in file_1_view_0.ProductName (products.csv, preserve row order)

Parameters:
- cap_i: file_0_view_0.Capacity, indexed by file_0_view_0.ShelfID
- val_j: file_1_view_0.Value, indexed by file_1_view_0.ProductName
- wt_j: file_1_view_0.Weight, indexed by file_1_view_0.ProductName

Decision Variables:
- x_{i,j}: Integer, ≥ 0, for each (i, j) ∈ 𝑆 × 𝑃

Objective:
- Maximize total value: sum over i ∈ 𝑆, j ∈ 𝑃 of file_1_view_0.Value × x_{i,j}

Constraints:
- For each i ∈ 𝑆: sum over j ∈ 𝑃 of file_1_view_0.Weight × x_{i,j} ≤ file_0_view_0.Capacity

All index sets, parameters, and constraints are mapped directly from the supplied tables as described above.