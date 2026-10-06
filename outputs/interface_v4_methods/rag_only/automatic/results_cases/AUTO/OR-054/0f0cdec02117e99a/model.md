ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of shelves (indexed by i), corresponding to ShelfID in file_0_view_0 (capacity.csv)
- 𝑃: Set of products (indexed by j), corresponding to ProductName in file_1_view_0 (products.csv)

Parameters:
- Capacity_i: Capacity of shelf i  
  Data Mapping: file_0_view_0, column "Capacity", keyed by "ShelfID"
- Value_j: Value per unit of product j  
  Data Mapping: file_1_view_0, column "Value", keyed by "ProductName"
- Weight_j: Weight per unit of product j  
  Data Mapping: file_1_view_0, column "Weight", keyed by "ProductName"

Decision Variables:
- x_{ij}: Number of units of product j placed on shelf i  
  Domain: nonnegative integers (x_{ij} ≥ 0, integer), ∀i ∈ 𝑆, ∀j ∈ 𝑃

Objective:
Maximize total value of products on all shelves:
\[
\max \sum_{i \in S} \sum_{j \in P} \text{Value}_j \cdot x_{ij}
\]

Constraints:
1. Shelf capacity constraints (for each shelf i ∈ 𝑆):
\[
\sum_{j \in P} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
2. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_+, \quad \forall i \in S, \forall j \in P
\]

Data Mapping:
- Shelves: file_0_view_0, "ShelfID"
- Shelf capacities: file_0_view_0, "Capacity"
- Products: file_1_view_0, "ProductName"
- Product values: file_1_view_0, "Value"
- Product weights: file_1_view_0, "Weight"

All indices, parameters, and constraints are explicitly aligned with the supplied data. No data is omitted or synthesized.