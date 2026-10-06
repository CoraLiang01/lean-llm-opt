ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let I be the set of products classified under ‘Organ’, indexed by i.

Parameters:
- Revenue_i: Revenue per unit of product i. (from SupermartGrocerySales-RetailAnalyticsDataset.csv, column: Revenue, table_id: file_0_view_0)
- InitialInventory_i: Initial inventory available for product i. (from SupermartGrocerySales-RetailAnalyticsDataset.csv, column: Initial Inventory, table_id: file_0_view_0)
- Demand_i: Deterministic demand for product i. (from SupermartGrocerySales-RetailAnalyticsDataset.csv, column: Demand, table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill, ∀i ∈ I. (Domain: integer, 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq InitialInventory_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq Demand_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I: All rows in SupermartGrocerySales-RetailAnalyticsDataset.csv (table_id: file_0_view_0) where Sub Category contains ‘Organ’.
- Revenue_i: SupermartGrocerySales-RetailAnalyticsDataset.csv, column: Revenue, table_id: file_0_view_0
- InitialInventory_i: SupermartGrocerySales-RetailAnalyticsDataset.csv, column: Initial Inventory, table_id: file_0_view_0
- Demand_i: SupermartGrocerySales-RetailAnalyticsDataset.csv, column: Demand, table_id: file_0_view_0