Abstract Optimization Model

Index Sets:
- Let 𝑃 be the set of products in the selected categories (electronics, apparel, homeware, etc.), as identified in table_id file_0_view_0, column Product Name.

Parameters:
- 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ₚ: Revenue per unit for product p ∈ 𝑃 (from file_0_view_0, column Revenue)
- 𝐷𝑒𝑚𝑎𝑛𝑑ₚ: Demand quantity for product p ∈ 𝑃 (from file_0_view_0, column Demand)
- 𝐼𝑛𝑖𝑡𝐼𝑛𝑣ₚ: Initial inventory for product p ∈ 𝑃 (from file_0_view_0, column Initial Inventory)

Decision Variables:
- 𝑥ₚ: Quantity of product p ∈ 𝑃 to fulfill (continuous, 0 ≤ 𝑥ₚ ≤ min{𝐷𝑒𝑚𝑎𝑛𝑑ₚ, 𝐼𝑛𝑖𝑡𝐼𝑛𝑣ₚ})

Objective:
- Maximize total revenue:
\[
\max \sum_{p \in P} \text{Revenue}_p \cdot x_p
\]

Constraints:
1. Inventory and demand fulfillment bounds for each product:
   - \( 0 \leq x_p \leq \min\{\text{Demand}_p, \text{Initial Inventory}_p\} \), ∀ p ∈ 𝑃

Data Mapping:
- Index set 𝑃 and all parameters (Revenueₚ, Demandₚ, Initial Inventoryₚ) are sourced from table_id file_0_view_0 in RetailSalesDataset.csv, using only records where Product Name contains "electronics", "apparel", or "homeware" (as per the validated filter applied by CSVQA).
- Revenue: column Revenue
- Demand: column Demand
- Initial Inventory: column Initial Inventory

No additional constraints or selection rules are imposed beyond those specified in the user query and the validated filter.