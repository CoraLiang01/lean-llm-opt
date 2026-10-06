Mathematical Optimization Model

Index Sets:
- 𝑰: Set of all products classified under ‘Fashion’  
  𝑰 = {i : i is a row in table_id = file_0_view_0}

Parameters (for each i ∈ 𝑰):
- 𝑟ᵢ: Revenue per unit of product i  
  (Source: file_0_view_0, column: Revenue)
- 𝑑ᵢ: Demand quantity for product i  
  (Source: file_0_view_0, column: Demand)
- 𝑠ᵢ: Initial inventory for product i  
  (Source: file_0_view_0, column: Initial Inventory)

Decision Variables:
- 𝑥ᵢ: Number of units of product i to fulfill  
  Domain: Integer, 0 ≤ 𝑥ᵢ ≤ min(𝑑ᵢ, 𝑠ᵢ), ∀ i ∈ 𝑰

Objective:
- Maximize total revenue from fulfilled ‘Fashion’ products:
\[
\max \sum_{i \in 𝑰} r_i x_i
\]

Constraints:
1. Demand and Inventory Fulfillment:
  For all i ∈ 𝑰:
\[
0 \leq x_i \leq \min(d_i, s_i)
\]

Data Mapping

- Index set 𝑰: All records in table_id = file_0_view_0 (SupermarketSales.csv filtered for ‘Fashion’ products)
- Parameter 𝑟ᵢ: file_0_view_0, column: Revenue
- Parameter 𝑑ᵢ: file_0_view_0, column: Demand
- Parameter 𝑠ᵢ: file_0_view_0, column: Initial Inventory

No additional constraints or data sources are used.