ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of ‘Aalop’ products (from RestaurantSalesreport.csv, table_id: file_0_view_0, column: Product Name)

Parameters:
- r_i: Revenue per unit of product i ∈ I (table_id: file_0_view_0, column: Revenue)
- d_i: Demand for product i ∈ I (table_id: file_0_view_0, column: Demand)
- s_i: Initial inventory of product i ∈ I (table_id: file_0_view_0, column: Initial Inventory)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min{d_i, s_i})

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
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I, parameters r_i, d_i, s_i are sourced from RestaurantSalesreport.csv (table_id: file_0_view_0), columns: Product Name, Revenue, Demand, Initial Inventory, filtered for ‘Aalop’ products.