ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of dairy products, indexed by i (from DairyGoodsSalesDataset.csv, column Full_Product_Name)

Parameters:
- D_i: Demand for product i (DairyGoodsSalesDataset.csv, column Demand)
- S_i: Initial inventory for product i (DairyGoodsSalesDataset.csv, column Initial Inventory)
- r_i: Revenue per unit of product i (DairyGoodsSalesDataset.csv, column Revenue)

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, x_i ≥ 0)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in \mathcal{I}} r_i \cdot x_i
  \]

Constraints:
1. Inventory and demand fulfillment bounds:
   \[
   0 \leq x_i \leq \min\{D_i, S_i\} \quad \forall i \in \mathcal{I}
   \]
   (x_i must not exceed either available inventory or demand for each product)

2. Integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{I}
   \]

Data Mapping:
- Index set 𝑰, parameters D_i, S_i, r_i are sourced from DairyGoodsSalesDataset.csv (table_id: file_0_view_0):
    - Product identifier: Full_Product_Name
    - Demand: Demand
    - Initial inventory: Initial Inventory
    - Revenue per unit: Revenue

No additional constraints or conditional bounds are specified or implied by the query. All bounds are unconditional and apply to every product in the dataset.