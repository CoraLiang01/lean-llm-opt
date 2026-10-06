ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of dairy products, indexed by i. (Corresponds to all Full_Product_Name in table_id: file_0_view_0)

Parameters:
- D_i: Demand for product i. (Demand, table_id: file_0_view_0)
- S_i: Initial inventory for product i. (Initial Inventory, table_id: file_0_view_0)
- r_i: Revenue per unit of product i. (Revenue, table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, 0 ≤ x_i ≤ min{D_i, S_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in \mathcal{I}} r_i \cdot x_i
  \]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq S_i \quad \forall i \in \mathcal{I}
   \]
2. Demand constraint:
   \[
   x_i \leq D_i \quad \forall i \in \mathcal{I}
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0,\quad x_i \in \mathbb{Z} \quad \forall i \in \mathcal{I}
   \]

Data Mapping:
- Index set 𝑰: All values in Full_Product_Name from table_id: file_0_view_0 (DairyGoodsSalesDataset.csv)
- Parameter D_i: Demand column, table_id: file_0_view_0
- Parameter S_i: Initial Inventory column, table_id: file_0_view_0
- Parameter r_i: Revenue column, table_id: file_0_view_0
- Decision variable x_i: defined for each i ∈ 𝑰

No data values or record counts are included; all references are symbolic and mapped to their exact sources.