Mathematical Model (Abstract Formulation)

Index Sets:
- 𝑰: Set of cabinets (CabinetID from file_0_view_0)
- 𝑱: Set of coffee products (ProductName from file_1_view_0)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity of cabinet i (from file_0_view_0, column Capacity)
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value per unit of product j (from file_1_view_0, column Value)
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Weight per unit of product j (from file_1_view_0, column Weight)

Decision Variables:
- 𝑥ᵢⱼ: Number of units of product j to place in cabinet i (integer, ≥ 0)

Objective:
Maximize total value across all cabinets:
\[
\max \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} \text{Value}_j \cdot x_{ij}
\]

Subject to:
1. Cabinet capacity constraints (for each cabinet i ∈ 𝑰):
\[
\sum_{j \in \mathcal{J}} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
2. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{I},\ j \in \mathcal{J}
\]

Data Mapping

Index Sets:
- 𝑰: All CabinetID values from file_0_view_0 (capacity.csv)
- 𝑱: All ProductName values from file_1_view_0 (products.csv)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0, column "Capacity", indexed by CabinetID
- 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0, column "Value", indexed by ProductName
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0, column "Weight", indexed by ProductName

Decision Variables:
- 𝑥ᵢⱼ: Integer, ≥ 0, indexed by (CabinetID, ProductName)

All data is mapped directly from the supplied tables and columns as specified above.