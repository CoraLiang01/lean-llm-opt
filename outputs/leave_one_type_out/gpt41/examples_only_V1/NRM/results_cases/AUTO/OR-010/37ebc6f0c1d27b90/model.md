ABSTRACT MATHEMATICAL OPTIMIZATION MODEL

Index Sets:
- 𝑃: Set of all products (indexed by i), where each product i is identified by the "Product Name" column in table_id file_0_view_0.

Parameters:
- 𝑟ᵢ: Per-unit revenue for product i ("Revenue" column, table_id file_0_view_0)
- 𝑑ᵢ: Expected demand for product i ("Demand" column, table_id file_0_view_0)
- 𝑠ᵢ: Initial inventory for product i ("Initial Inventory" column, table_id file_0_view_0)

Decision Variables:
- 𝑥ᵢ: Number of units of product i fulfilled (integer, 𝑥ᵢ ≥ 0), ∀i ∈ 𝑃

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} r_i x_i
  \]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in P
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in P
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in P
   \]

Data Mapping:
- Index set 𝑃 and all parameters (𝑟ᵢ, 𝑑ᵢ, 𝑠ᵢ) are sourced from table_id file_0_view_0 in "MobileSalesDataset.csv":
    - Product identifier: "Product Name"
    - Per-unit revenue: "Revenue"
    - Demand: "Demand"
    - Initial inventory: "Initial Inventory"