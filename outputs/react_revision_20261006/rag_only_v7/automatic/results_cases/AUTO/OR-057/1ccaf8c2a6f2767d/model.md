Mathematical Optimization Model

Index Sets:
- 𝑃: Set of platforms (PlatformID from file_0_view_0)
- 𝐺: Set of games (ProductName from file_1_view_0)

Parameters:
- 𝐶ₚ: Memory capacity of platform p ∈ 𝑃 (Capacity from file_0_view_0)
- 𝑉g: Value of game g ∈ 𝐺 (Value from file_1_view_0)
- 𝑊g: Memory requirement (weight) of game g ∈ 𝐺 (Weight from file_1_view_0)

Decision Variables:
- x_{p,g}: Number of units of game g ∈ 𝐺 to be listed on platform p ∈ 𝑃 (integer, x_{p,g} ≥ 0)

Objective:
Maximize total value of games listed across all platforms:
\[
\max \sum_{p \in 𝑃} \sum_{g \in 𝐺} 𝑉_g \cdot x_{p,g}
\]

Subject to:

1. Platform memory capacity constraints:
\[
\sum_{g \in 𝐺} 𝑊_g \cdot x_{p,g} \leq 𝐶_p \quad \forall p \in 𝑃
\]

2. Nonnegativity and integrality:
\[
x_{p,g} \in \mathbb{Z}_+, \quad \forall p \in 𝑃,\, g \in 𝐺
\]

Data Mapping

- 𝑃 (platforms): file_0_view_0.PlatformID
- 𝐺 (games): file_1_view_0.ProductName
- 𝐶ₚ: file_0_view_0.Capacity (indexed by PlatformID)
- 𝑉g: file_1_view_0.Value (indexed by ProductName)
- 𝑊g: file_1_view_0.Weight (indexed by ProductName)
- Decision variable x_{p,g}: number of units of game g (file_1_view_0.ProductName) to list on platform p (file_0_view_0.PlatformID), integer and ≥ 0

All index sets, parameters, and constraints are mapped directly from the supplied tables as described above.