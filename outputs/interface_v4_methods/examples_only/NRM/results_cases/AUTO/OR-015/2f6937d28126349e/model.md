ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of products classified under ‘Aalop’ (from RestaurantSalesreport.csv, [Product Name] with prefix 'Aalop', table_id: file_0_view_0)

Parameters:
- r_i: Revenue per unit of product i ∈ I (from [Revenue], table_id: file_0_view_0)
- d_i: Demand for product i ∈ I during the sales horizon (from [Demand], table_id: file_0_view_0)
- s_i: Initial inventory of product i ∈ I (from [Initial Inventory], table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min{s_i, d_i})

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
- Index set I, and parameters r_i, d_i, s_i are sourced from RestaurantSalesreport.csv (table_id: file_0_view_0), using columns:
  - [Product Name] (prefix 'Aalop') → I
  - [Revenue] → r_i
  - [Demand] → d_i
  - [Initial Inventory] → s_i