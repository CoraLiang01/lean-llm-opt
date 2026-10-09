ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of products classified under ‘Organ’ (from Sub Category column where value contains or starts with "Organ").

Parameters:
- 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Revenue per unit of product i ∈ 𝑰 (from Revenue column, table_id: file_0_view_0).
- 𝐼𝐼ᵢ: Initial inventory of product i ∈ 𝑰 (from Initial Inventory column, table_id: file_0_view_0).
- 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: Demand quantity for product i ∈ 𝑰 (from Demand column, table_id: file_0_view_0).

Decision Variables:
- 𝑥ᵢ: Number of units of product i ∈ 𝑰 to fulfill (continuous or integer, as appropriate; domain: 0 ≤ 𝑥ᵢ ≤ min{𝐼𝐼ᵢ, 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ}).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ \cdot 𝑥ᵢ
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   0 \leq 𝑥ᵢ \leq 𝐼𝐼ᵢ \quad \forall i \in 𝑰
   \]
2. Demand fulfillment constraint for each product:
   \[
   0 \leq 𝑥ᵢ \leq 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰 and all parameters are sourced from table_id: file_0_view_0 in SupermartGrocerySales-RetailAnalyticsDataset.csv:
    - ‘Sub Category’ (for identifying ‘Organ’ products)
    - ‘Revenue’ (for 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ)
    - ‘Initial Inventory’ (for 𝐼𝐼ᵢ)
    - ‘Demand’ (for 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ)