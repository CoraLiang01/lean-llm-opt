ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑰: Set of cabinets (CabinetID from file_0_view_0 in capacity.csv)
- 𝑱: Set of coffee products (ProductName from file_1_view_0 in products.csv)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity of cabinet i (file_0_view_0, column: Capacity, indexed by CabinetID)
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value per unit of product j (file_1_view_0, column: Value, indexed by ProductName)
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Weight per unit of product j (file_1_view_0, column: Weight, indexed by ProductName)

Decision Variables:
- 𝑥ᵢⱼ: Number of units of product j to place in cabinet i (integer, 𝑥ᵢⱼ ≥ 0, ∀ i ∈ 𝑰, j ∈ 𝑱)

Objective:
Maximize total value across all cabinets:
\[
\max \sum_{i \in 𝑰} \sum_{j \in 𝑱} 𝑉𝑎𝑙𝑢𝑒ⱼ \cdot 𝑥ᵢⱼ
\]

Subject to:
- Cabinet capacity constraints (for each cabinet i):
\[
\sum_{j \in 𝑱} 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ \cdot 𝑥ᵢⱼ \leq 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ \quad \forall i \in 𝑰
\]
- Nonnegativity and integrality:
\[
𝑥ᵢⱼ \in \mathbb{Z}_{\geq 0} \quad \forall i \in 𝑰, j \in 𝑱
\]

DATA MAPPING

- 𝑰 (CabinetID): file_0_view_0, column CabinetID
- 𝑱 (ProductName): file_1_view_0, column ProductName
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0, column Capacity, indexed by CabinetID
- 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0, column Value, indexed by ProductName
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0, column Weight, indexed by ProductName

All cabinets and products are included as indexed by their respective business IDs. Each cabinet’s constraint uses its own capacity and the weights of all products. All decision variables are integer and nonnegative.