Mathematical Optimization Model

Index Sets:
- 𝑰: Set of display areas (DisplayID from file_0_view_0)
- 𝑱: Set of boat types (ProductName from file_1_view_0)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity of display area i (file_0_view_0, Capacity, indexed by DisplayID)
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value of boat type j (file_1_view_0, Value, indexed by ProductName)
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Size of boat type j (file_1_view_0, Weight, indexed by ProductName)

Decision Variables:
- 𝑥ᵢⱼ: Number of units of boat type j to place in display area i (integer, ≥ 0)

Objective:
Maximize total value of boats displayed:
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
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{I},\ j \in \mathcal{J}
\]

Data Mapping:
- 𝑰 (Display areas): file_0_view_0, DisplayID
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0, Capacity (indexed by DisplayID)
- 𝑱 (Boat types): file_1_view_0, ProductName
- 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0, Value (indexed by ProductName)
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0, Weight (indexed by ProductName)
- Decision variables 𝑥ᵢⱼ: indexed by (DisplayID, ProductName) pairs

No additional constraints or data are imposed beyond those mapped above.