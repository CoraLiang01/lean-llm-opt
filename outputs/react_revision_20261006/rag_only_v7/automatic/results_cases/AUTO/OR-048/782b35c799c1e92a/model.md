Mathematical Model (Abstract Formulation)

Index Sets:
- 𝑆: Set of storage areas (indexed by i), with StorageID from file_0_view_0.
- 𝑃: Set of air conditioner product types (indexed by j), with ProductName from file_1_view_0.

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity of storage area i (from file_0_view_0, column Capacity).
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value of one unit of product type j (from file_1_view_0, column Value).
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Size (weight) of one unit of product type j (from file_1_view_0, column Weight).

Decision Variables:
- 𝑥ᵢⱼ: Number of units of product type j to place in storage area i (integer, 𝑥ᵢⱼ ≥ 0).

Objective:
Maximize total value of air conditioners allocated:
\[
\max \sum_{i \in S} \sum_{j \in P} \text{Value}_j \cdot x_{ij}
\]

Subject to:
- Storage area capacity constraints (for each i ∈ S):
\[
\sum_{j \in P} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
- Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_+, \quad \forall i \in S,\, j \in P
\]

Data Mapping

Index Sets:
- 𝑆: All StorageID in file_0_view_0 (capacity.csv), column StorageID.
- 𝑃: All ProductName in file_1_view_0 (products.csv), column ProductName.

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0, column Capacity, indexed by StorageID.
- 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0, column Value, indexed by ProductName.
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0, column Weight, indexed by ProductName.

Decision Variables:
- 𝑥ᵢⱼ: Integer, ≥ 0, for each (StorageID i, ProductName j).

Objective:
- Maximize total value: sum over i (StorageID) and j (ProductName) of Value_j * x_ij.

Constraints:
- For each StorageID i: sum over j (ProductName) of Weight_j * x_ij ≤ Capacity_i.

All mappings use the exact column and table_id names as above.