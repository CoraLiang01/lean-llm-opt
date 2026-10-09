ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all ‘TABLET’ smartphone models, indexed by i.

Parameters:
- Revenue_i: Revenue per unit for model i. (Source: SmartphoneRetailOutletSalesData.csv, column 'Revenue', table_id: file_0_view_0)
- InitialInventory_i: Initial inventory available for model i. (Source: SmartphoneRetailOutletSalesData.csv, column 'Initial Inventory', table_id: file_0_view_0)
- Demand_i: Deterministic demand for model i. (Source: SmartphoneRetailOutletSalesData.csv, column 'Demand', table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of model i to fulfill, integer, with 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i}.

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each model:
   \[
   x_i \leq InitialInventory_i \quad \forall i \in I
   \]
2. Demand constraint for each model:
   \[
   x_i \leq Demand_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I and all parameters (Revenue_i, InitialInventory_i, Demand_i) are sourced from SmartphoneRetailOutletSalesData.csv (table_id: file_0_view_0), using only rows where 'Product Name' has prefix 'TABLET'. Columns used: 'Product Name' (for i), 'Revenue', 'Initial Inventory', 'Demand'.