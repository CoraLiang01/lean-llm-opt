ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of shelves, indexed by i. (from file_0_view_0, column ShelfID)
- 𝑃: Set of products, indexed by j. (from file_1_view_0, column ProductName)

Parameters:
- Capacity_i: Capacity of shelf i ∈ 𝑆.  
  Data Mapping: file_0_view_0, column Capacity, key: ShelfID
- Value_j: Value per unit of product j ∈ 𝑃.  
  Data Mapping: file_1_view_0, column Value, key: ProductName
- Weight_j: Weight per unit of product j ∈ 𝑃.  
  Data Mapping: file_1_view_0, column Weight, key: ProductName

Decision Variables:
- x_{ij}: Number of units of product j allocated to shelf i  
  Domain: x_{ij} ∈ ℤ₊ (nonnegative integers), ∀ i ∈ 𝑆, j ∈ 𝑃

Objective:
Maximize total value of products allocated to all shelves:
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
x_{ij} \in \mathbb{Z}_+, \quad \forall i \in S, j \in P
\]

Data Mapping:
- 𝑆 (shelves): file_0_view_0, column ShelfID
- 𝑃 (products): file_1_view_0, column ProductName
- Capacity_i: file_0_view_0, column Capacity, key: ShelfID
- Value_j: file_1_view_0, column Value, key: ProductName
- Weight_j: file_1_view_0, column Weight, key: ProductName

This model maximizes the total value of products displayed, ensuring that the total weight on each shelf does not exceed its capacity, with integer allocation decisions for each product-shelf pair.