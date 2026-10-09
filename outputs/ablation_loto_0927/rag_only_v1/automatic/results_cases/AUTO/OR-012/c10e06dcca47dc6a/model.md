Abstract Optimization Model

Index Sets:
- 𝑃: Set of products, indexed by i.

Parameters:
- 𝑟𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Revenue per unit of product i. (from Revenue)
- 𝑑𝑒𝑚𝑎𝑛𝑑ᵢ: Deterministic demand for product i. (from Demand)
- 𝑖𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦ᵢ: Initial inventory available for product i. (from Initial Inventory)

Decision Variables:
- 𝑥ᵢ: Number of units of product i to fulfill for customer purchases (integer, 0 ≤ 𝑥ᵢ ≤ min{𝑑𝑒𝑚𝑎𝑛𝑑ᵢ, 𝑖𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦ᵢ})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} \text{revenue}_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq \text{inventory}_i \quad \forall i \in P
   \]
2. Demand constraint for each product:
   \[
   x_i \leq \text{demand}_i \quad \forall i \in P
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in P
   \]

Data Mapping:
- Table: OnlineSalesDataset.csv (table_id: file_0_view_0)
  - Product index set P: Product Name
  - Parameter revenueᵢ: Revenue
  - Parameter inventoryᵢ: Initial Inventory
  - Parameter demandᵢ: Demand