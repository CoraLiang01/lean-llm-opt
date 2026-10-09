Mathematical Optimization Model

Index Sets:
- 𝑰: Set of display areas (DisplayID from file_0_view_0)
- 𝑱: Set of vessel types (ProductName from file_1_view_0)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity of display area i (from file_0_view_0, column Capacity)
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value of vessel type j (from file_1_view_0, column Value)
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Size (weight) of vessel type j (from file_1_view_0, column Weight)

Decision Variables:
- 𝑥ᵢⱼ: Number of vessels of type j to place in display area i (integer, 𝑥ᵢⱼ ≥ 0)

Objective:
Maximize total value of vessels displayed:
\[
\max \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} \text{Value}_j \cdot x_{ij}
\]

Constraints:
1. Capacity constraint for each display area:
\[
\sum_{j \in \mathcal{J}} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \quad \forall i \in \mathcal{I}
\]
2. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I},\ j \in \mathcal{J}
\]

Data Mapping:
- 𝑰 (Display areas): file_0_view_0, column DisplayID
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0, column Capacity, keyed by DisplayID
- 𝑱 (Vessel types): file_1_view_0, column ProductName
- 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0, column Value, keyed by ProductName
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0, column Weight, keyed by ProductName
- Decision variable 𝑥ᵢⱼ: indexed by (DisplayID, ProductName)

This model maximizes the total value of boats assigned to display areas, subject to each area's capacity, using the provided data.