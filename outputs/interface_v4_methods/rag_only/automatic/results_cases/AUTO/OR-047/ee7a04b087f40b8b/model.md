ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑃: Set of platforms (PlatformId from file_0_view_0)
- 𝐺: Set of game genres (ProductName from file_1_view_0)

Parameters:
- 𝐶ₚ: Memory capacity of platform p ∈ 𝑃 (Capacity from file_0_view_0, table_id: file_0_view_0, column: Capacity, key: PlatformId)
- 𝑉g: Value per unit of genre g ∈ 𝐺 (Value from file_1_view_0, table_id: file_1_view_0, column: Value, key: ProductName)
- 𝑊g: Memory requirement per unit of genre g ∈ 𝐺 (Weight from file_1_view_0, table_id: file_1_view_0, column: Weight, key: ProductName)

Decision Variables:
- x_{p,g}: Number of units of games from genre g to be listed on platform p (integer, x_{p,g} ≥ 0)

Objective:
Maximize total value across all platforms:
\[
\max \sum_{p \in 𝑃} \sum_{g \in 𝐺} 𝑉_g \cdot x_{p,g}
\]

Constraints:
1. Platform memory capacity:
\[
\forall p \in 𝑃: \quad \sum_{g \in 𝐺} 𝑊_g \cdot x_{p,g} \leq 𝐶_p
\]
2. Nonnegativity and integrality:
\[
\forall p \in 𝑃,\, g \in 𝐺: \quad x_{p,g} \in \mathbb{Z}_{\geq 0}
\]

Data Mapping:
- 𝑃 (platforms): PlatformId from file_0_view_0 (capacity.csv)
- 𝐺 (genres): ProductName from file_1_view_0 (products.csv)
- 𝐶ₚ: file_0_view_0, column Capacity, key PlatformId
- 𝑉g: file_1_view_0, column Value, key ProductName
- 𝑊g: file_1_view_0, column Weight, key ProductName

All parameters and sets are mapped directly from the supplied files as described above.