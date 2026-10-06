ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑃: Set of platforms (indexed by i), from capacity.csv [file_0_view_0, column PlatformID]
- 𝐺: Set of games (indexed by j), from products.csv [file_1_view_0, column ProductName]

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Memory capacity of platform i  
  Data Mapping: file_0_view_0, column Capacity, key PlatformID
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value of game j  
  Data Mapping: file_1_view_0, column Value, key ProductName
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Memory requirement (weight) of game j  
  Data Mapping: file_1_view_0, column Weight, key ProductName

Decision Variables:
- 𝑥ᵢⱼ: Number of units of game j to be listed on platform i  
  Domain: Integer, 𝑥ᵢⱼ ≥ 0

Objective:
Maximize total value of games listed across all platforms:
\[
\max \sum_{i \in P} \sum_{j \in G} \text{Value}_j \cdot x_{ij}
\]

Constraints:
1. Platform memory capacity: For each platform i ∈ P,
\[
\sum_{j \in G} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]

2. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in P,\, j \in G
\]

Data Mapping:
- Platform set and capacities: file_0_view_0, columns PlatformID, Capacity
- Game set, values, and weights: file_1_view_0, columns ProductName, Value, Weight

All indices, parameters, and constraints are directly mapped from the supplied data. No additional constraints or synthesized IDs are introduced.