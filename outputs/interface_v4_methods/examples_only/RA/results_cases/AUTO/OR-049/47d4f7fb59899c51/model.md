ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of shelves (ShelfID from file_0_view_0)
- 𝑃: Set of products (ProductName from file_1_view_0)

Parameters:
- cap_s: Capacity of shelf s ∈ 𝑆
- val_p: Value of product p ∈ 𝑃
- w_p: Weight of product p ∈ 𝑃

Decision Variables:
- x_{s,p} ∈ ℤ₊: Number of units of product p placed on shelf s (for all s ∈ 𝑆, p ∈ 𝑃)

Objective:
Maximize ∑_{s∈𝑆} ∑_{p∈𝑃} val_p · x_{s,p}

Constraints:
1. Shelf capacity constraints (for all s ∈ 𝑆):
  ∑_{p∈𝑃} w_p · x_{s,p} ≤ cap_s

2. Nonnegativity and integrality:
  x_{s,p} ∈ {0, 1, 2, ...} for all s ∈ 𝑆, p ∈ 𝑃

DATA MAPPING

- 𝑆 (shelf index): file_0_view_0.ShelfID
- 𝑃 (product index): file_1_view_0.ProductName
- cap_s: file_0_view_0.Capacity (for shelf s)
- val_p: file_1_view_0.Value (for product p)
- w_p: file_1_view_0.Weight (for product p)