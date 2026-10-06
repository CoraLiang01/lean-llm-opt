ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of displays (ShelfID from file_0_view_0)
- 𝑃: Set of products (ProductName from file_1_view_0)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ₛ: Capacity of display s ∈ 𝑆 (Capacity from file_0_view_0, indexed by ShelfID)
- 𝑉𝑎𝑙𝑢𝑒ₚ: Value per unit of product p ∈ 𝑃 (Value from file_1_view_0, indexed by ProductName)
- 𝑊𝑒𝑖𝑔ℎ𝑡ₚ: Weight per unit of product p ∈ 𝑃 (Weight from file_1_view_0, indexed by ProductName)
- 𝑝₁: The first product in file_1_view_0 (ProductName at source_row 0)

Decision Variables:
- 𝑥ₛₚ: Number of units of product p ∈ 𝑃 placed on display s ∈ 𝑆 (nonnegative integer)

Objective:
Maximize total value of all products placed:
\[
\max \sum_{s \in S} \sum_{p \in P} \text{Value}_p \cdot x_{sp}
\]

Subject to:

1. Display capacity constraints (for each display s ∈ 𝑆):
\[
\sum_{p \in P} \text{Weight}_p \cdot x_{sp} \leq \text{Capacity}_s
\]

2. Minimum total quantity of the first product (𝑝₁) across all displays:
\[
\sum_{s \in S} x_{s p_1} \geq 5
\]

3. Nonnegativity and integrality:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

DATA MAPPING

- 𝑆 (displays): file_0_view_0.ShelfID (original row order)
- 𝑃 (products): file_1_view_0.ProductName (original row order)
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ₛ: file_0_view_0.Capacity, indexed by ShelfID
- 𝑉𝑎𝑙𝑢𝑒ₚ: file_1_view_0.Value, indexed by ProductName
- 𝑊𝑒𝑖𝑔ℎ𝑡ₚ: file_1_view_0.Weight, indexed by ProductName
- 𝑝₁: file_1_view_0.ProductName at source_row 0

All indices and parameters must use the exact values and order as returned in the CSVQA_DATA.