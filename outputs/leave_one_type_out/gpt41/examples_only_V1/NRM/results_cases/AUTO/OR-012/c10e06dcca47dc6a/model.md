Abstract Mathematical Optimization Model

Index Sets:
- 𝑃: Set of products, indexed by i.

Parameters:
- 𝑟𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Revenue per unit of product i. [Source: file_0_view_0, column 'Revenue']
- 𝑖𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦ᵢ: Initial inventory of product i. [Source: file_0_view_0, column 'Initial Inventory']
- 𝑑𝑒𝑚𝑎𝑛𝑑ᵢ: Deterministic demand for product i. [Source: file_0_view_0, column 'Demand']

Decision Variables:
- 𝑥ᵢ: Number of units of product i to fulfill (integer, 0 ≤ 𝑥ᵢ ≤ min{𝑖𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦ᵢ, 𝑑𝑒𝑚𝑎𝑛𝑑ᵢ})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} \text{revenue}_i \cdot x_i
  \]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq \text{inventory}_i \quad \forall i \in P
   \]
2. Demand constraint:
   \[
   x_i \leq \text{demand}_i \quad \forall i \in P
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in P
   \]

Data Mapping:
- file_0_view_0, column 'Product Name': Product identifier (index set P)
- file_0_view_0, column 'Revenue': Parameter revenueᵢ
- file_0_view_0, column 'Initial Inventory': Parameter inventoryᵢ
- file_0_view_0, column 'Demand': Parameter demandᵢ

This model maximizes total revenue by choosing, for each product, the number of units to fulfill, subject to inventory and demand limits.