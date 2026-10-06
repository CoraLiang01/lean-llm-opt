ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑰: Set of display areas (DisplayID from capacity.csv, table_id: file_0_view_0)
- 𝑱: Set of boat types (ProductName from products.csv, table_id: file_1_view_0)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity of display area i ∈ 𝑰  
  Data Mapping: file_0_view_0, column "Capacity", key "DisplayID"
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value of one unit of boat type j ∈ 𝑱  
  Data Mapping: file_1_view_0, column "Value", key "ProductName"
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Size of one unit of boat type j ∈ 𝑱  
  Data Mapping: file_1_view_0, column "Weight", key "ProductName"

Decision Variables:
- 𝑥_{ij}: Number of units of boat type j ∈ 𝑱 to place in display area i ∈ 𝑰  
  Domain: Integer, 𝑥_{ij} ≥ 0

Objective:
Maximize total value of boats displayed:
\[
\max \sum_{i \in 𝑰} \sum_{j \in 𝑱} 𝑉𝑎𝑙𝑢𝑒ⱼ \cdot 𝑥_{ij}
\]

Constraints:
1. Capacity constraint for each display area:
\[
\sum_{j \in 𝑱} 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ \cdot 𝑥_{ij} \leq 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ \quad \forall i \in 𝑰
\]
2. Nonnegativity and integrality:
\[
𝑥_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in 𝑰,\, j \in 𝑱
\]

DATA MAPPING

- 𝑰 (Display Areas): file_0_view_0, column "DisplayID"
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0, column "Capacity", key "DisplayID"
- 𝑱 (Boat Types): file_1_view_0, column "ProductName"
- 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0, column "Value", key "ProductName"
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0, column "Weight", key "ProductName"