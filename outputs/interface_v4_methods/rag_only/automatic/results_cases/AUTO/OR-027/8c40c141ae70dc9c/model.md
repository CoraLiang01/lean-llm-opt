ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let I be the set of products classified as ‘Organ’ (from SupermartGrocerySales-RetailAnalyticsDataset.csv, column: Sub Category, filtered to contain "Organ").

Parameters:
- r_i: Revenue per unit of product i ∈ I (table_id: file_0_view_0, column: Revenue)
- s_i: Initial inventory of product i ∈ I (table_id: file_0_view_0, column: Initial Inventory)
- d_i: Demand for product i ∈ I (table_id: file_0_view_0, column: Demand)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (continuous or integer, as appropriate; domain: 0 ≤ x_i ≤ min{s_i, d_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i x_i
  \]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in I
   \]

Data Mapping:
- Index set I: All rows in SupermartGrocerySales-RetailAnalyticsDataset.csv (table_id: file_0_view_0) where Sub Category contains "Organ"
- r_i: Revenue (file_0_view_0, column: Revenue)
- s_i: Initial Inventory (file_0_view_0, column: Initial Inventory)
- d_i: Demand (file_0_view_0, column: Demand)