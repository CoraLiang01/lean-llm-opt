ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑃: Set of bread products (ProductName from file_1_view_0)

Parameters:
- 𝑣ₚ: Expected profit per unit of product p ∈ 𝑃 (Value from file_1_view_0, column Value)
- 𝑤ₚ: Storage weight per unit of product p ∈ 𝑃 (Weight from file_1_view_0, column Weight)
- 𝐶: Total available storage capacity (Capacity from file_0_view_0, column Capacity)

Decision Variables:
- 𝑥ₚ: Number of units of product p ∈ 𝑃 to order each day (integer, 𝑥ₚ ≥ 0)

Objective:
Maximize total expected profit:
\[
\max \sum_{p \in 𝑃} vₚ \cdot xₚ
\]

Subject to:
- Storage capacity constraint:
\[
\sum_{p \in 𝑃} wₚ \cdot xₚ \leq C
\]
- Nonnegativity and integrality:
\[
xₚ \in \mathbb{Z}_{\geq 0} \quad \forall p \in 𝑃
\]

DATA MAPPING

- 𝑃: All ProductName values from file_1_view_0 (products.csv, column ProductName)
- 𝑣ₚ: file_1_view_0, column Value, keyed by ProductName
- 𝑤ₚ: file_1_view_0, column Weight, keyed by ProductName
- 𝐶: file_0_view_0, column Capacity (capacity.csv, single value)