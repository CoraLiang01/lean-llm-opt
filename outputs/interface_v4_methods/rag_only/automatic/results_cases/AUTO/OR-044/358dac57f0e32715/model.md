ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of sections (SectionID from file_0_view_0)
- 𝑃: Set of products (ProductName from file_1_view_0)

Parameters:
- 𝐶ₛ: Capacity (display space limit) of section s ∈ 𝑆
- 𝑉ₚ: Price (revenue per unit) of product p ∈ 𝑃
- 𝑊ₚ: Shelf space required per unit of product p ∈ 𝑃

Decision Variables:
- 𝑥ₛₚ: Number of units of product p ∈ 𝑃 to stock in section s ∈ 𝑆 (integer, 𝑥ₛₚ ≥ 0)

Objective:
Maximize total revenue:
\[
\max \sum_{s \in S} \sum_{p \in P} V_p \cdot x_{sp}
\]

Subject to:

Section capacity constraints (for each section s ∈ 𝑆):
\[
\sum_{p \in P} W_p \cdot x_{sp} \leq C_s
\]

Nonnegativity and integrality:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

DATA MAPPING

- 𝑆 (sections): SectionID from file_0_view_0 (capacity.csv)
- 𝑃 (products): ProductName from file_1_view_0 (products.csv)
- 𝐶ₛ: file_0_view_0, column "Capacity", indexed by SectionID
- 𝑉ₚ: file_1_view_0, column "Value", indexed by ProductName
- 𝑊ₚ: file_1_view_0, column "Weight", indexed by ProductName

All parameters and indices are mapped directly from the supplied tables, preserving original row order and identifiers.