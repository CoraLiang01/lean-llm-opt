ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of shelves (indexed by i), corresponding to ShelfID from file_0_view_0 (capacity.csv)
- 𝑃: Set of products (indexed by j), corresponding to ProductName from file_1_view_0 (products.csv)

Parameters:
- cap_i: Capacity of shelf i  
  Data Mapping: cap_i = file_0_view_0[Capacity] where file_0_view_0[ShelfID] = i
- val_j: Value per unit of product j  
  Data Mapping: val_j = file_1_view_0[Value] where file_1_view_0[ProductName] = j
- wt_j: Weight per unit of product j  
  Data Mapping: wt_j = file_1_view_0[Weight] where file_1_view_0[ProductName] = j

Decision Variables:
- x_{i,j}: Number of units of product j to place on shelf i  
  Domain: x_{i,j} ∈ ℤ₊ (nonnegative integers), ∀ i ∈ 𝑆, j ∈ 𝑃

Objective:
Maximize total value of products placed on all shelves:
\[
\max \sum_{i \in 𝑆} \sum_{j \in 𝑃} val_j \cdot x_{i,j}
\]

Constraints:
1. Shelf capacity constraints (for each shelf i ∈ 𝑆):
\[
\sum_{j \in 𝑃} wt_j \cdot x_{i,j} \leq cap_i
\]
2. Nonnegativity and integrality:
\[
x_{i,j} \in \mathbb{Z}_+, \quad \forall i \in 𝑆,\, j \in 𝑃
\]

Data Mapping:
- 𝑆 = {file_0_view_0[ShelfID] for all rows in file_0_view_0}
- 𝑃 = {file_1_view_0[ProductName] for all rows in file_1_view_0}
- cap_i: file_0_view_0[Capacity] for shelf i
- val_j: file_1_view_0[Value] for product j
- wt_j: file_1_view_0[Weight] for product j

Each shelf and product is indexed by its explicit business identifier (ShelfID, ProductName) as provided in the source files. No sorting or re-indexing is performed. All variables are integer as required.