ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of shelves (indexed by i), corresponding to ShelfID in file_0_view_0 (capacity.csv)
- 𝑃: Set of products (indexed by j), corresponding to ProductName in file_1_view_0 (products.csv)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦ᵢ: Capacity of shelf i  
  Data Mapping: file_0_view_0, column "Capacity", key "ShelfID"
- 𝑉𝑎𝑙𝑢𝑒ⱼ: Value per unit of product j  
  Data Mapping: file_1_view_0, column "Value", key "ProductName"
- 𝑊𝑒𝑖𝑔ℎ𝑡ⱼ: Weight per unit of product j  
  Data Mapping: file_1_view_0, column "Weight", key "ProductName"

Decision Variables:
- 𝑥ᵢⱼ: Number of units of product j to place on shelf i  
  Domain: integer, 𝑥ᵢⱼ ≥ 0, ∀ i ∈ 𝑆, j ∈ 𝑃

Objective:
Maximize total value of products placed on all shelves:
\[
\max \sum_{i \in S} \sum_{j \in P} \text{Value}_j \cdot x_{ij}
\]

Constraints:
1. Shelf capacity constraints (for each shelf i ∈ S):
\[
\sum_{j \in P} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]

2. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, j \in P
\]

Data Mapping:
- Shelves: file_0_view_0, "ShelfID"
- Shelf capacities: file_0_view_0, "Capacity"
- Products: file_1_view_0, "ProductName"
- Product values: file_1_view_0, "Value"
- Product weights: file_1_view_0, "Weight"

Each shelf's allocation is constrained by its own capacity, and all variables are indexed by both shelf and product as required.