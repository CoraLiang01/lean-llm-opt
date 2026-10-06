ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑰: Set of cabinets (CabinetID from file_0_view_0 in capacity.csv)
- 𝑱: Set of coffee products (ProductName from file_1_view_0 in products.csv)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity of cabinet i (from file_0_view_0, column Capacity, indexed by CabinetID)
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value per unit of product j (from file_1_view_0, column Value, indexed by ProductName)
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Weight per unit of product j (from file_1_view_0, column Weight, indexed by ProductName)

Decision Variables:
- 𝑥ᵢⱼ: Number of units of product j to place in cabinet i (integer, 𝑥ᵢⱼ ≥ 0, ∀ i ∈ 𝑰, j ∈ 𝑱)

Objective:
Maximize total value of products allocated:
\[
\max \sum_{i \in 𝑰} \sum_{j \in 𝑱} 𝑉𝑎𝑙𝑢𝑒ⱼ \cdot 𝑥ᵢⱼ
\]

Subject to:
- Cabinet capacity constraints (for each cabinet i ∈ 𝑰):
\[
\sum_{j \in 𝑱} 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ \cdot 𝑥ᵢⱼ \leq 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ
\]
- Integer and nonnegativity constraints:
\[
𝑥ᵢⱼ \in \mathbb{Z}_{\geq 0} \quad \forall i \in 𝑰, j \in 𝑱
\]

DATA MAPPING

Index Sets:
- 𝑰: CabinetID from file_0_view_0 (capacity.csv)
- 𝑱: ProductName from file_1_view_0 (products.csv)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0, column "Capacity", indexed by "CabinetID"
- 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0, column "Value", indexed by "ProductName"
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0, column "Weight", indexed by "ProductName"

Decision Variables:
- 𝑥ᵢⱼ: Number of units of product j (ProductName) to place in cabinet i (CabinetID), integer, ≥ 0

All cabinets and products are included as indexed by their respective business IDs.