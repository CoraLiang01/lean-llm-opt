ABSTRACT Optimization Model

Index Sets:
- Let I be the set of products classified under ‘Organ’, indexed by i.

Parameters:
- Revenue_i: Revenue per unit of product i. (from column ‘Revenue’)
- InitialInventory_i: Initial inventory of product i. (from column ‘Initial Inventory’)
- Demand_i: Demand quantity for product i. (from column ‘Demand’)

Decision Variables:
- x_i: Number of units of product i to fulfill, ∀i ∈ I. (x_i ≥ 0, integer or continuous as appropriate)

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
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in I
   \]

Data Mapping:
- Table: file_0_view_0 (SupermartGrocerySales-RetailAnalyticsDataset.csv)
  - Index set I: All rows where ‘Sub Category’ contains or starts with "Organ"
  - Revenue_i: column ‘Revenue’
  - InitialInventory_i: column ‘Initial Inventory’
  - Demand_i: column ‘Demand’