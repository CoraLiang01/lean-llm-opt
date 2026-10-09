Mathematical Model (Abstract Formulation)

Index Sets:
- 𝑰: Set of vehicle types (ProductName from file_1_view_0)

Parameters:
- 𝑣𝑎𝑙𝑢𝑒ᵢ: Benefit coefficient of vehicle type i ∈ 𝑰 (Value from file_1_view_0, column "Value")
- 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ: Inventory weight/unit of vehicle type i ∈ 𝑰 (Weight from file_1_view_0, column "Weight")
- 𝐶: Total inventory capacity (Capacity from file_0_view_0, column "Capacity")

Decision Variables:
- 𝑥ᵢ: Number of units of vehicle type i ∈ 𝑰 to order daily (integer, 𝑥ᵢ ≥ 0)

Objective:
Maximize total benefit:
\[
\max \sum_{i \in 𝑰} 𝑣𝑎𝑙𝑢𝑒ᵢ \cdot 𝑥ᵢ
\]

Subject to:
- Inventory capacity constraint:
\[
\sum_{i \in 𝑰} 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ \cdot 𝑥ᵢ \leq 𝐶
\]
- Integrality and nonnegativity:
\[
𝑥ᵢ \in \mathbb{Z}_+, \quad \forall i \in 𝑰
\]

Data Mapping

- Index set 𝑰: All ProductName values from file_1_view_0 (products.csv)
- Parameter 𝑣𝑎𝑙𝑢𝑒ᵢ: file_1_view_0, column "Value", keyed by "ProductName"
- Parameter 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ: file_1_view_0, column "Weight", keyed by "ProductName"
- Parameter 𝐶: file_0_view_0, column "Capacity"
- Decision variable 𝑥ᵢ: defined for each i ∈ 𝑰 (ProductName from file_1_view_0)