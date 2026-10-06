ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑃: Set of platforms (indexed by i), from file_0_view_0.PlatformId
- 𝐺: Set of game genres (indexed by j), from file_1_view_0.ProductName

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Memory capacity of platform i (from file_0_view_0.Capacity)
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value per unit of genre j (from file_1_view_0.Value)
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Memory required per unit of genre j (from file_1_view_0.Weight)

Decision Variables:
- 𝑥ᵢⱼ: Number of units of games from genre j to be listed on platform i (integer, ≥ 0)

Objective:
Maximize total value across all platforms and genres:
\[
\max \sum_{i \in P} \sum_{j \in G} \text{Value}_j \cdot x_{ij}
\]

Constraints:
1. Platform memory capacity:
\[
\sum_{j \in G} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \quad \forall i \in P
\]
2. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in P,\, j \in G
\]

DATA MAPPING

- Index set 𝑃: All file_0_view_0.PlatformId
- Index set 𝐺: All file_1_view_0.ProductName
- Parameter 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: file_0_view_0.Capacity, keyed by file_0_view_0.PlatformId
- Parameter 𝑉𝑎𝑙𝑢𝑒ⱼ: file_1_view_0.Value, keyed by file_1_view_0.ProductName
- Parameter 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: file_1_view_0.Weight, keyed by file_1_view_0.ProductName
- Decision variable 𝑥ᵢⱼ: integer, indexed by (file_0_view_0.PlatformId, file_1_view_0.ProductName)