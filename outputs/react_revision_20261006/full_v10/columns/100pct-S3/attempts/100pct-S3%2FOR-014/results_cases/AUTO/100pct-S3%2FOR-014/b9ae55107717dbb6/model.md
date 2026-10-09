Mathematical Model

Index Sets:
- Let S be the set of shelves, with elements indexed by s and ShelfID from file_0_view_0.
- Let P be the set of products, with elements indexed by p and ProductName from file_1_view_0.

Parameters:
- Capacity_s: Capacity of shelf s (from file_0_view_0, column Capacity, key ShelfID)
- Value_p: Value per unit of product p (from file_1_view_0, column Value, key ProductName)
- Weight_p: Weight per unit of product p (from file_1_view_0, column Weight, key ProductName)

Decision Variables:
- x_{s,p}: Number of units of product p allocated to shelf s
  - Domain: x_{s,p} ∈ ℤ_{\geq 0} (nonnegative integers), ∀ s ∈ S, p ∈ P

Objective:
Maximize total value across all shelves:
$$
\max \sum_{s \in S} \sum_{p \in P} \text{Value}_p \cdot x_{s,p}
$$

Constraints:
1. Shelf capacity constraints (for each shelf s):
$$
\sum_{p \in P} \text{Weight}_p \cdot x_{s,p} \leq \text{Capacity}_s, \quad \forall s \in S
$$

2. Nonnegativity and integrality:
$$
x_{s,p} \in \mathbb{Z}_{\geq 0}, \quad \forall s \in S,\, p \in P
$$

Data Mapping

- S (shelf index): file_0_view_0, column ShelfID
- P (product index): file_1_view_0, column ProductName
- Capacity_s: file_0_view_0, columns ShelfID and Capacity (map by ShelfID)
- Value_p: file_1_view_0, columns ProductName and Value (map by ProductName)
- Weight_p: file_1_view_0, columns ProductName and Weight (map by ProductName)
- Decision variable x_{s,p}: number of units of product p (ProductName) on shelf s (ShelfID)

All parameters and index sets are defined directly from the current CSV data as described above.