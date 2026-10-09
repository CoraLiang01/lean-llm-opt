Mathematical Optimization Model

Index Sets:
- 𝑆: Set of shelves (ShelfID from file_0_view_0)
- 𝑃: Set of products (ProductName from file_1_view_0)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ₛ: Capacity of shelf s ∈ 𝑆 (file_0_view_0, column: Capacity, key: ShelfID)
- 𝑉𝑎𝑙𝑢𝑒ₚ: Value of product p ∈ 𝑃 (file_1_view_0, column: Value, key: ProductName)
- 𝑊𝑒𝑖𝑔ℎ𝑡ₚ: Weight of product p ∈ 𝑃 (file_1_view_0, column: Weight, key: ProductName)

Decision Variables:
- 𝑥ₛₚ: Number of units of product p ∈ 𝑃 to place on shelf s ∈ 𝑆 (integer, 𝑥ₛₚ ≥ 0)

Objective:
Maximize total value of products displayed:
\[
\max \sum_{s \in S} \sum_{p \in P} \text{Value}_p \cdot x_{sp}
\]

Constraints:
1. Shelf capacity constraints (for each shelf s ∈ 𝑆):
\[
\sum_{p \in P} \text{Weight}_p \cdot x_{sp} \leq \text{Capacity}_s
\]
2. Nonnegativity and integrality:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

Data Mapping:
- 𝑆 (shelf index): file_0_view_0, column ShelfID
- 𝑃 (product index): file_1_view_0, column ProductName
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ₛ: file_0_view_0, columns ShelfID, Capacity
- 𝑉𝑎𝑙𝑢𝑒ₚ: file_1_view_0, columns ProductName, Value
- 𝑊𝑒𝑖𝑔ℎ𝑡ₚ: file_1_view_0, columns ProductName, Weight
- Decision variables 𝑥ₛₚ indexed by (ShelfID, ProductName) pairs

All index sets, parameters, and constraints are defined directly from the supplied data.