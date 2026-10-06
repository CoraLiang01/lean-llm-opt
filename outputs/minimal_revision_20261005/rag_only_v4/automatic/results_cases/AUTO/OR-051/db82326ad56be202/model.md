ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑰: Set of cabinets (CabinetID from file_0_view_0)
- 𝑱: Set of coffee products (ProductName from file_1_view_0)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity of cabinet i (from file_0_view_0, column Capacity)
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value per unit of product j (from file_1_view_0, column Value)
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Weight per unit of product j (from file_1_view_0, column Weight)

Decision Variables:
- 𝑥ᵢⱼ: Number of units of product j allocated to cabinet i (integer, 𝑥ᵢⱼ ≥ 0)

Objective:
Maximize total value of products allocated:
\[
\max \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} \text{Value}_j \cdot x_{ij}
\]

Subject to:

1. Cabinet capacity constraints (for all i ∈ 𝑰):
\[
\sum_{j \in \mathcal{J}} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]

2. Nonnegativity and integrality (for all i ∈ 𝑰, j ∈ 𝑱):
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

DATA MAPPING

- 𝑰 (cabinets): All CabinetID values from file_0_view_0, column CabinetID
- 𝑱 (products): All ProductName values from file_1_view_0, column ProductName
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0, columns CabinetID (for i), Capacity
- 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0, columns ProductName (for j), Value
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0, columns ProductName (for j), Weight