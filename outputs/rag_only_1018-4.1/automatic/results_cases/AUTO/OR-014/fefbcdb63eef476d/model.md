Abstract Optimization Model

Index Sets:
- 𝑃: Set of pizza types, indexed by i.
  (Data Mapping: Pizza types are identified by "Product Name" in table_id "file_0_view_0".)

Parameters:
- 𝑟ᵢ: Revenue per unit of pizza type i.
  (Data Mapping: "Revenue" column in table_id "file_0_view_0".)
- 𝑑ᵢ: Demand for pizza type i.
  (Data Mapping: "Demand" column in table_id "file_0_view_0".)
- 𝑠ᵢ: Initial inventory available for pizza type i.
  (Data Mapping: "Initial Inventory" column in table_id "file_0_view_0".)

Decision Variables:
- 𝑥ᵢ: Number of units of pizza type i to fulfill (integer, 𝑥ᵢ ≥ 0).

Objective:
- Maximize total revenue:
  max ∑_{i ∈ 𝑃} 𝑟ᵢ · 𝑥ᵢ

Constraints:
1. Inventory and demand fulfillment bounds for each pizza type:
  0 ≤ 𝑥ᵢ ≤ min{𝑑ᵢ, 𝑠ᵢ}  ∀ i ∈ 𝑃

2. Integrality:
  𝑥ᵢ ∈ ℤ₊  ∀ i ∈ 𝑃

Data Mapping:
- Index set 𝑃: All unique values in "Product Name" from table_id "file_0_view_0" (PizzaSalesDataset.csv).
- Parameter 𝑟ᵢ: "Revenue" column in table_id "file_0_view_0".
- Parameter 𝑑ᵢ: "Demand" column in table_id "file_0_view_0".
- Parameter 𝑠ᵢ: "Initial Inventory" column in table_id "file_0_view_0".
- All records from table_id "file_0_view_0" are used without additional filtering.

This model determines the optimal integer fulfillment quantities for each pizza type to maximize total revenue, subject to both inventory and demand constraints.