ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of sections (SectionID from file_0_view_0)
- 𝑃: Set of products (ProductName from file_1_view_0)

Parameters:
- 𝐶ₛ: Capacity of section s ∈ 𝑆 (file_0_view_0, column: Capacity, key: SectionID)
- 𝑉ₚ: Price (revenue per unit) of product p ∈ 𝑃 (file_1_view_0, column: Value, key: ProductName)
- 𝑊ₚ: Shelf space required per unit of product p ∈ 𝑃 (file_1_view_0, column: Weight, key: ProductName)

Decision Variables:
- 𝑥ₛₚ: Number of units of product p to stock in section s (integer, 𝑥ₛₚ ≥ 0)

Objective:
Maximize total revenue:
\[
\max \sum_{s \in S} \sum_{p \in P} V_p \cdot x_{sp}
\]

Constraints:
1. Section capacity constraints (for each section s ∈ 𝑆):
\[
\sum_{p \in P} W_p \cdot x_{sp} \leq C_s
\]
2. Nonnegativity and integrality:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

DATA MAPPING

- Section set S and parameter Cₛ: file_0_view_0, SectionID and Capacity columns
- Product set P and parameters Vₚ, Wₚ: file_1_view_0, ProductName, Value, and Weight columns
- Decision variables xₛₚ are indexed by SectionID (from file_0_view_0) and ProductName (from file_1_view_0)

No data was omitted or synthesized. All resource dimensions and business identifiers are preserved.