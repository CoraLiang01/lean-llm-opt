ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑰: Set of display areas (DisplayID from file_0_view_0)
- 𝑱: Set of vessel types (ProductName from file_1_view_0)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity of display area i  
  Data Mapping: file_0_view_0, columns: DisplayID, Capacity
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value of vessel type j  
  Data Mapping: file_1_view_0, columns: ProductName, Value
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Size (weight) of vessel type j  
  Data Mapping: file_1_view_0, columns: ProductName, Weight

Decision Variables:
- 𝑥ᵢⱼ: Number of vessels of type j to place in display area i  
  Domain: nonnegative integers (𝑥ᵢⱼ ∈ ℤ₊)

Objective:
Maximize total value of vessels displayed:
\[
\max \sum_{i \in 𝑰} \sum_{j \in 𝑱} 𝑉𝑎𝑙𝑢𝑒ⱼ \cdot 𝑥ᵢⱼ
\]

Constraints:
1. Capacity constraint for each display area:
\[
\sum_{j \in 𝑱} 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ \cdot 𝑥ᵢⱼ \leq 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ \quad \forall i \in 𝑰
\]
2. Nonnegativity and integrality:
\[
𝑥ᵢⱼ \in \mathbb{Z}_+, \quad \forall i \in 𝑰,\, j \in 𝑱
\]

Data Mapping:
- 𝑰 (Display areas): file_0_view_0, DisplayID
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0, Capacity (indexed by DisplayID)
- 𝑱 (Vessel types): file_1_view_0, ProductName
- 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0, Value (indexed by ProductName)
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0, Weight (indexed by ProductName)

All indices, parameters, and mappings are directly aligned with the supplied data. No data was omitted or synthesized.