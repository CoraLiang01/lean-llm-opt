ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let I be the set of products classified under ‘Organ’, indexed by i. (From table_id: file_0_view_0, column: Sub Category, filtered by Sub Category prefix "Organic")

Parameters:
- Revenue_i: Revenue per unit of product i (table_id: file_0_view_0, column: Revenue)
- Inventory_i: Initial inventory of product i (table_id: file_0_view_0, column: Initial Inventory)
- Demand_i: Demand quantity for product i (table_id: file_0_view_0, column: Demand)

Decision Variables:
- x_i: Number of units of product i to fulfill (continuous or integer, as appropriate; x_i ≥ 0)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq Inventory_i \quad \forall i \in I
   \]
2. Demand constraint for each product:
   \[
   x_i \leq Demand_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in I
   \]

Data Mapping:
- Index set I: All records in table_id: file_0_view_0, column: Sub Category, where Sub Category has prefix "Organic"
- Revenue_i: table_id: file_0_view_0, column: Revenue
- Inventory_i: table_id: file_0_view_0, column: Initial Inventory
- Demand_i: table_id: file_0_view_0, column: Demand

Note: The model uses only the records returned by the validated filter (Sub Category prefix "Organic") as per the query. No additional filtering or selection is applied.