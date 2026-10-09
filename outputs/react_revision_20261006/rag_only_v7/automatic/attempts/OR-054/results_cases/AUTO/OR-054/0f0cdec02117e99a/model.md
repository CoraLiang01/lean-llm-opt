Mathematical Optimization Model

Index Sets:
- 𝑆: Set of shelves (ShelfID from file_0_view_0)
- 𝑃: Set of products (ProductName from file_1_view_0)

Parameters:
- 𝐶_𝑖: Capacity of shelf 𝑖 ∈ 𝑆 (Capacity from file_0_view_0)
- 𝑉_𝑗: Value per unit of product 𝑗 ∈ 𝑃 (Value from file_1_view_0)
- 𝑊_𝑗: Weight per unit of product 𝑗 ∈ 𝑃 (Weight from file_1_view_0)

Decision Variables:
- 𝑥_{𝑖𝑗}: Number of units of product 𝑗 ∈ 𝑃 placed on shelf 𝑖 ∈ 𝑆 (𝑥_{𝑖𝑗} ∈ ℤ₊)

Objective:
Maximize total value of products on all shelves:
\[
\max \sum_{i \in S} \sum_{j \in P} V_j \cdot x_{ij}
\]

Constraints:
1. Shelf capacity constraints (for each shelf 𝑖 ∈ 𝑆):
\[
\sum_{j \in P} W_j \cdot x_{ij} \leq C_i
\]
2. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_+, \quad \forall i \in S,\, j \in P
\]

Data Mapping:
- 𝑆 (shelves): file_0_view_0, column ShelfID
- 𝑃 (products): file_1_view_0, column ProductName
- 𝐶_𝑖: file_0_view_0, column Capacity, key ShelfID
- 𝑉_𝑗: file_1_view_0, column Value, key ProductName
- 𝑊_𝑗: file_1_view_0, column Weight, key ProductName