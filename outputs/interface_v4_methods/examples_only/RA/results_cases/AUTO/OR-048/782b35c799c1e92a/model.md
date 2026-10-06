ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of storage areas (indexed by i), from file_0_view_0, column StorageID.
- 𝑃: Set of air conditioner product types (indexed by j), from file_1_view_0, column ProductName.

Parameters:
- cap_i: Capacity of storage area i.  
  Data Mapping: file_0_view_0, column Capacity, key StorageID.
- val_j: Value of one unit of product type j.  
  Data Mapping: file_1_view_0, column Value, key ProductName.
- size_j: Size (weight) of one unit of product type j.  
  Data Mapping: file_1_view_0, column Weight, key ProductName.

Decision Variables:
- x_{i,j}: Number of units of product type j allocated to storage area i.  
  Domain: x_{i,j} ∈ ℤ₊ (nonnegative integers), ∀ i ∈ 𝑆, j ∈ 𝑃.

Objective:
Maximize total value of air conditioners allocated:
\[
\max \sum_{i \in S} \sum_{j \in P} val_j \cdot x_{i,j}
\]

Constraints:
1. Storage area capacity constraints (for each storage area i):
\[
\sum_{j \in P} size_j \cdot x_{i,j} \leq cap_i, \quad \forall i \in S
\]

2. Nonnegativity and integrality:
\[
x_{i,j} \in \mathbb{Z}_+, \quad \forall i \in S,\, j \in P
\]

DATA MAPPING

- Storage areas (i ∈ S): file_0_view_0, StorageID (original row order preserved)
- cap_i: file_0_view_0, Capacity, key StorageID
- Product types (j ∈ P): file_1_view_0, ProductName (original row order preserved)
- val_j: file_1_view_0, Value, key ProductName
- size_j: file_1_view_0, Weight, key ProductName

No additional constraints or relationships are imposed beyond those described above.