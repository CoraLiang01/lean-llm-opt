Mathematical Model

Index Sets:
- Let S be the set of storage areas, indexed by s, with StorageID from file_0_view_0.
- Let P be the set of air conditioner types, indexed by p, with ProductName from file_1_view_0.

Parameters:
- cap_s: Capacity of storage area s. (file_0_view_0, column: Capacity, key: StorageID)
- val_p: Value of one unit of air conditioner type p. (file_1_view_0, column: Value, key: ProductName)
- w_p: Size (Weight) of one unit of air conditioner type p. (file_1_view_0, column: Weight, key: ProductName)

Decision Variables:
- x_{s,p}: Number of units of air conditioner type p allocated to storage area s.
  Domain: x_{s,p} ∈ ℤ₊ (nonnegative integers), ∀ s ∈ S, p ∈ P

Objective:
Maximize total value of allocated air conditioners:
  maximize ∑_{s ∈ S} ∑_{p ∈ P} val_p · x_{s,p}

Constraints:
1. Storage area capacity constraints:
  For each storage area s ∈ S:
    ∑_{p ∈ P} w_p · x_{s,p} ≤ cap_s

2. Nonnegativity and integrality:
  x_{s,p} ∈ ℤ₊, ∀ s ∈ S, p ∈ P

Data Mapping

Index Sets:
- S: file_0_view_0, column StorageID
- P: file_1_view_0, column ProductName

Parameters:
- cap_s: file_0_view_0, column Capacity, key StorageID
- val_p: file_1_view_0, column Value, key ProductName
- w_p: file_1_view_0, column Weight, key ProductName

Decision Variables:
- x_{s,p}: Number of units of product p in storage area s (indexed by StorageID and ProductName)

All parameters and index sets are mapped directly from the specified columns and keys in the returned CSV data.