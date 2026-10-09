Mathematical Optimization Model

Index Sets:
- 𝑰: Set of all car models classified under ‘FDK57’ (from file_0_view_0, column "Product Name", one entry per returned row).

Parameters:
- 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Revenue per unit for car model i ∈ 𝑰 (file_0_view_0, "Revenue")
- 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: Demand quantity for car model i ∈ 𝑰 (file_0_view_0, "Demand")
- 𝐼𝑛𝑣ᵢ: Initial inventory for car model i ∈ 𝑰 (file_0_view_0, "Initial Inventory")

Decision Variables:
- 𝑥ᵢ: Number of units of car model i ∈ 𝑰 to fulfill (integer, 0 ≤ 𝑥ᵢ ≤ min{𝐷𝑒𝑚𝑎𝑛𝑑ᵢ, 𝐼𝑛𝑣ᵢ})

Objective:
- Maximize total revenue:
  max ∑_{i ∈ 𝑰} 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ · 𝑥ᵢ

Constraints:
1. Demand fulfillment:  𝑥ᵢ ≤ 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ  ∀ i ∈ 𝑰
2. Inventory limit:    𝑥ᵢ ≤ 𝐼𝑛𝑣ᵢ      ∀ i ∈ 𝑰
3. Non-negativity and integrality: 𝑥ᵢ ∈ {0, 1, ..., min{𝐷𝑒𝑚𝑎𝑛𝑑ᵢ, 𝐼𝑛𝑣ᵢ}} ∀ i ∈ 𝑰

Data Mapping:
- Index set 𝑰: All rows in file_0_view_0 where "Product Name" has prefix "FDK57"
- 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: file_0_view_0, column "Revenue"
- 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: file_0_view_0, column "Demand"
- 𝐼𝑛𝑣ᵢ: file_0_view_0, column "Initial Inventory"