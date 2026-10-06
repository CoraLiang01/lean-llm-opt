ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑃: Set of dairy products, indexed by i.

Parameters:
- D_i: Demand for product i (from DairyGoodsSalesDataset.csv, column Demand, table_id: file_0_view_0)
- I_i: Initial inventory for product i (from DairyGoodsSalesDataset.csv, column Initial Inventory, table_id: file_0_view_0)
- R_i: Revenue per unit for product i (from DairyGoodsSalesDataset.csv, column Revenue, table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, 0 ≤ x_i ≤ min{D_i, I_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} R_i \cdot x_i
  \]

Constraints:
1. Inventory limit:
   \[
   x_i \leq I_i \quad \forall i \in P
   \]
2. Demand limit:
   \[
   x_i \leq D_i \quad \forall i \in P
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in P
   \]
   (If x_i must be integer, add: x_i ∈ ℤ_+)

Data Mapping:
- DairyGoodsSalesDataset.csv (table_id: file_0_view_0)
    - Product index: Full_Product_Name
    - Demand: Demand
    - Initial inventory: Initial Inventory
    - Revenue per unit: Revenue

This model maximizes total revenue by optimally fulfilling orders for each dairy product, subject to both inventory and demand constraints.