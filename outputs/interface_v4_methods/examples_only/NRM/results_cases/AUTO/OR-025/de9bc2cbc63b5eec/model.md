ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all TABLET smartphone models, indexed by i. (Source: SmartphoneRetailOutletSalesData.csv, Product Name, filtered where Product Name has prefix 'TABLET')

Parameters:
- Revenue_i: Revenue per unit for model i. (Source: SmartphoneRetailOutletSalesData.csv, Revenue)
- InitialInventory_i: Initial inventory available for model i. (Source: SmartphoneRetailOutletSalesData.csv, Initial Inventory)
- Demand_i: Deterministic demand for model i. (Source: SmartphoneRetailOutletSalesData.csv, Demand)

Decision Variables:
- x_i: Number of units of TABLET model i to fulfill (integer, 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i})

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
- Index set I, and all parameters (Revenue_i, InitialInventory_i, Demand_i) are sourced from SmartphoneRetailOutletSalesData.csv, table_id: file_0_view_0, columns: Product Name (filtered for prefix 'TABLET'), Revenue, Initial Inventory, Demand.