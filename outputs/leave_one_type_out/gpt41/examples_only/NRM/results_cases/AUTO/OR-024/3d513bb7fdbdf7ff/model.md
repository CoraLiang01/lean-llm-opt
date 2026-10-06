ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of products with Product Name starting with 'S700_' (from table_id: file_0_view_0, column: Product Name)

Parameters:
- r_i: Revenue per unit of product i ∈ I (from table_id: file_0_view_0, column: Revenue)
- inv_i: Initial Inventory of product i ∈ I (from table_id: file_0_view_0, column: Initial Inventory)
- d_i: Demand for product i ∈ I (from table_id: file_0_view_0, column: Demand)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min(inv_i, d_i))

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq inv_i \quad \forall i \in I
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0,\ x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I, and all parameters (r_i, inv_i, d_i) are sourced from table_id: file_0_view_0, columns: Product Name, Revenue, Initial Inventory, Demand, respectively.