ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑃: Set of dairy products, indexed by i.

Parameters:
- D_i: Demand for product i. (from DairyGoodsSalesDataset.csv, column 'Demand')
- I_i: Initial inventory for product i. (from DairyGoodsSalesDataset.csv, column 'Initial Inventory')
- R_i: Revenue per unit of product i. (from DairyGoodsSalesDataset.csv, column 'Revenue')

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, 0 ≤ x_i ≤ min{D_i, I_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} R_i \cdot x_i
  \]

Constraints:
1. Inventory limit for each product:
   \[
   x_i \leq I_i \quad \forall i \in P
   \]
2. Demand limit for each product:
   \[
   x_i \leq D_i \quad \forall i \in P
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0,\ x_i \in \mathbb{Z} \quad \forall i \in P
   \]

Data Mapping:
- Index set 𝑃 and all parameters (D_i, I_i, R_i) are sourced from DairyGoodsSalesDataset.csv (table_id: file_0_view_0):
    - Product identifier: 'Full_Product_Name'
    - Demand: 'Demand'
    - Initial Inventory: 'Initial Inventory'
    - Revenue per unit: 'Revenue'