ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑃: Set of products/categories (indexed by i), where each i corresponds to a unique 'Product Name' in table_id file_0_view_0.

Parameters:
- Revenue_i: Revenue per unit for product i. Source: file_0_view_0, column 'Revenue'.
- InitialInventory_i: Initial inventory available for product i. Source: file_0_view_0, column 'Initial Inventory'.
- Demand_i: Demand for product i. Source: file_0_view_0, column 'Demand'.

Decision Variables:
- x_i: Quantity of product i to fulfill (continuous, x_i ≥ 0).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq InitialInventory_i \quad \forall i \in P
   \]
2. Demand constraint for each product:
   \[
   x_i \leq Demand_i \quad \forall i \in P
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in P
   \]

Data Mapping:
- Index set 𝑃, and all parameters (Revenue_i, InitialInventory_i, Demand_i) are sourced from table_id file_0_view_0 in RetailSalesDataset.csv:
    - Product Name → index set 𝑃
    - Revenue → Revenue_i
    - Initial Inventory → InitialInventory_i
    - Demand → Demand_i