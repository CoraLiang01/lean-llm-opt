Abstract Mathematical Optimization Model

Index Sets:
- Let 𝑃 be the set of all products, indexed by i.

Parameters:
- 𝑟ᵢ: Revenue per unit for product i ∈ 𝑃. (from column Revenue)
- 𝑑ᵢ: Expected demand for product i ∈ 𝑃 during the sales cycle. (from column Demand)
- 𝑠ᵢ: Initial inventory available for product i ∈ 𝑃. (from column Initial Inventory)

Decision Variables:
- 𝑥ᵢ ∈ ℤ₊: Number of units of product i fulfilled (orders fulfilled), for each i ∈ 𝑃.

Objective:
- Maximize total revenue from fulfilled orders:
  \[
  \max \sum_{i \in P} r_i x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in P
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in P
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in P
   \]

Data Mapping:
- All parameters are sourced from MobileSalesDataset.csv (table_id: file_0_view_0):
    - Product index set 𝑃: Product Name
    - Revenue per unit 𝑟ᵢ: Revenue
    - Demand 𝑑ᵢ: Demand
    - Initial Inventory 𝑠ᵢ: Initial Inventory
- No filters were applied; all 71 product records in file_0_view_0 are included directly.

This model maximizes total revenue by optimally allocating available inventory to fulfill deterministic demand for each product, subject to inventory and demand limits, with no restocking or in-transit inventory.