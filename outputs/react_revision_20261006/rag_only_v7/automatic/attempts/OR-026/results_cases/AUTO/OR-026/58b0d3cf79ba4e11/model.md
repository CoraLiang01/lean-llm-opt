Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- 𝑰: Set of all products i classified as 'Fashion' (i.e., all products in "file_0_view_0" where "Product Name" contains 'Fashion').

Parameters:
- r_i: Revenue per unit of product i ∈ 𝑰. [from "file_0_view_0", column "Revenue"]
- d_i: Deterministic demand for product i ∈ 𝑰. [from "file_0_view_0", column "Demand"]
- s_i: Initial inventory for product i ∈ 𝑰. [from "file_0_view_0", column "Initial Inventory"]

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill. (Domain: integer, 0 ≤ x_i ≤ min{d_i, s_i})

Objective:
- Maximize total revenue from fulfilled 'Fashion' products:
  maximize ∑_{i ∈ 𝑰} r_i x_i

Constraints:
1. Demand fulfillment constraint:
  x_i ≤ d_i  ∀ i ∈ 𝑰
2. Inventory constraint:
  x_i ≤ s_i  ∀ i ∈ 𝑰
3. Non-negativity and integrality:
  x_i ∈ {0, 1, ..., min{d_i, s_i}} ∀ i ∈ 𝑰

Data Mapping:
- Index set 𝑰: All rows in "file_0_view_0" from "SupermarketSales.csv" where "Product Name" contains 'Fashion'.
- r_i: "Revenue" column in "file_0_view_0" for i ∈ 𝑰.
- d_i: "Demand" column in "file_0_view_0" for i ∈ 𝑰.
- s_i: "Initial Inventory" column in "file_0_view_0" for i ∈ 𝑰.
- Decision variable x_i: defined for each i ∈ 𝑰.

This model maximizes total revenue from 'Fashion' products, subject to deterministic demand and available inventory for each product.