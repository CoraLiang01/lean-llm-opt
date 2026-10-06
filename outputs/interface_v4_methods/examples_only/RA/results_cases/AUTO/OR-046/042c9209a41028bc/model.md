ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑃: Set of products, indexed by i (corresponds to ProductName in products.csv)

Parameters:
- v_i: Profit (income) per unit of product i  
  Data Mapping: file_1_view_0, column 'Value', key 'ProductName'
- w_i: Space (weight) consumed per unit of product i  
  Data Mapping: file_1_view_0, column 'Weight', key 'ProductName'
- C: Total available stock capacity  
  Data Mapping: file_0_view_0, column 'Capacity'

Decision Variables:
- x_i: Number of units of product i to order each day (x_i ∈ ℤ₊, i.e., nonnegative integers), ∀i ∈ 𝑃

Objective:
Maximize total profit:
\[
\max \sum_{i \in P} v_i x_i
\]

Constraints:
1. Stock capacity constraint:
\[
\sum_{i \in P} w_i x_i \leq C
\]
2. Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in P
\]

Data Mapping:
- ProductName (file_1_view_0) identifies each product i.
- v_i: file_1_view_0, column 'Value', key 'ProductName'
- w_i: file_1_view_0, column 'Weight', key 'ProductName'
- C: file_0_view_0, column 'Capacity'

All products and the single overall stock capacity are included as required. No additional constraints or variables are introduced.