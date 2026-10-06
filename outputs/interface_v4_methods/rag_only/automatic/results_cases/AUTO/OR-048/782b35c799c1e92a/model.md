ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of storage areas (indexed by i), from file_0_view_0, column StorageID.
- 𝑃: Set of air conditioner product types (indexed by j), from file_1_view_0, column ProductName.

Parameters:
- cap_i: Capacity of storage area i ∈ 𝑆.
  Data Mapping: cap_i = CSVQA_DATA["file_0_view_0"]["Capacity"][StorageID = i]
- val_j: Value of one unit of product type j ∈ 𝑃.
  Data Mapping: val_j = CSVQA_DATA["file_1_view_0"]["Value"][ProductName = j]
- size_j: Size (weight) of one unit of product type j ∈ 𝑃.
  Data Mapping: size_j = CSVQA_DATA["file_1_view_0"]["Weight"][ProductName = j]

Decision Variables:
- x_{i,j}: Number of units of product type j allocated to storage area i.
  Domain: x_{i,j} ∈ ℤ₊ (nonnegative integers), ∀ i ∈ 𝑆, j ∈ 𝑃

Objective:
Maximize total value of air conditioners allocated:
\[
\max \sum_{i \in S} \sum_{j \in P} val_j \cdot x_{i,j}
\]

Constraints:
1. Storage area capacity constraints (for each storage area i ∈ 𝑆):
\[
\sum_{j \in P} size_j \cdot x_{i,j} \leq cap_i \qquad \forall i \in S
\]

2. Nonnegativity and integrality:
\[
x_{i,j} \in \mathbb{Z}_+, \qquad \forall i \in S, j \in P
\]

Data Mapping:
- Storage areas: file_0_view_0, StorageID
- Storage area capacities: file_0_view_0, Capacity
- Product types: file_1_view_0, ProductName
- Product values: file_1_view_0, Value
- Product sizes: file_1_view_0, Weight

No additional constraints or relationships are imposed beyond those described above. Each storage area receives its own capacity constraint, and all decision variables are indexed by both storage area and product type, as required.