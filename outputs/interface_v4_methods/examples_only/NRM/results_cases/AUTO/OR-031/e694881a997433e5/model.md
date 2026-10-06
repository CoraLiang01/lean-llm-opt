ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑃: Set of dairy products, indexed by i.

Parameters:
- 𝑑𝑒𝑚𝑎𝑛𝑑_i: Demand for product i. (from DairyGoodsSalesDataset.csv, column: Demand)
- 𝑖𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦_i: Initial Inventory for product i. (from DairyGoodsSalesDataset.csv, column: Initial Inventory)
- 𝑝𝑟𝑖𝑐𝑒_i: Revenue per unit for product i. (from DairyGoodsSalesDataset.csv, column: Revenue)

Decision Variables:
- 𝑥_i ≥ 0: Number of units of product i to fulfill.

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} \text{price}_i \cdot x_i
  \]

Constraints:
1. Inventory limit:
   \[
   x_i \leq \text{inventory}_i \quad \forall i \in P
   \]
2. Demand limit:
   \[
   x_i \leq \text{demand}_i \quad \forall i \in P
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in P
   \]

Data Mapping:
- Index set 𝑃 and all parameters are sourced from DairyGoodsSalesDataset.csv (table_id: file_0_view_0):
    - Product identifier: Full_Product_Name
    - Initial Inventory: Initial Inventory
    - Demand: Demand
    - Revenue per unit: Revenue