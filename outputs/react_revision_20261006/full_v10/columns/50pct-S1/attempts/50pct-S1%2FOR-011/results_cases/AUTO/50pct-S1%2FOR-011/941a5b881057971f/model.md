Mathematical Model

Index Sets:
- Let P be the set of products, with each product identified by ProductName from file_1_view_0.

Parameters:
- v_p: Value per unit of product p ∈ P, from Value in file_1_view_0.
- w_p: Weight per unit of product p ∈ P, from Weight in file_1_view_0.
- C: Overall stock capacity, from Capacity in file_0_view_0.

Decision Variables:
- x_p: Number of units of product p ∈ P to order each day. Domain: x_p ∈ ℤ≥0.

Objective:
Maximize total benefit:
maximize ∑_{p ∈ P} v_p x_p

Subject to:
Overall stock capacity constraint:
  ∑_{p ∈ P} w_p x_p ≤ C

Nonnegativity and integrality:
  x_p ∈ ℤ≥0  ∀ p ∈ P

Data Mapping

Index Sets:
- P: ProductName in file_1_view_0

Parameters:
- v_p: Value column in file_1_view_0, keyed by ProductName
- w_p: Weight column in file_1_view_0, keyed by ProductName
- C: Capacity column in file_0_view_0

Decision Variables:
- x_p: Number of units of product p ∈ P to order each day (nonnegative integer)