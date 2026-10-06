ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of dairy products (indexed by i)

Parameters:
- 𝑑𝑒𝑚𝑎𝑛𝑑ᵢ: Demand for product i (from column Demand)
- 𝑖𝑛𝑣ᵢ: Initial inventory for product i (from column Initial Inventory)
- 𝑝ᵢ: Revenue per unit of product i (from column Revenue)

Decision Variables:
- 𝑥ᵢ: Number of units of product i to fulfill (integer, 0 ≤ 𝑥ᵢ ≤ min{𝑑𝑒𝑚𝑎𝑛𝑑ᵢ, 𝑖𝑛𝑣ᵢ})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} pᵢ \cdot xᵢ
  \]

Constraints:
1. Inventory limit for each product:
   \[
   xᵢ \leq 𝑖𝑛𝑣ᵢ \quad \forall i \in 𝑰
   \]
2. Demand fulfillment cannot exceed demand:
   \[
   xᵢ \leq 𝑑𝑒𝑚𝑎𝑛𝑑ᵢ \quad \forall i \in 𝑰
   \]
3. Non-negativity and integrality:
   \[
   xᵢ \geq 0, \quad xᵢ \in \mathbb{Z} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Table: DairyGoodsSalesDataset.csv (table_id: file_0_view_0)
  - Index set 𝑰: Full_Product_Name
  - Parameter 𝑑𝑒𝑚𝑎𝑛𝑑ᵢ: Demand
  - Parameter 𝑖𝑛𝑣ᵢ: Initial Inventory
  - Parameter 𝑝ᵢ: Revenue

This model maximizes total revenue by optimally fulfilling orders for each dairy product, subject to initial inventory and demand constraints.