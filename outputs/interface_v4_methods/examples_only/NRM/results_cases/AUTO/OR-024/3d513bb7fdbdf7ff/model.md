ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of products, where each product i ∈ I corresponds to a product with Product Name starting with 'S700_' (from table_id: file_0_view_0, column: Product Name).

Parameters:
- r_i: Revenue per unit of product i (from table_id: file_0_view_0, column: Revenue).
- s_i: Initial inventory of product i (from table_id: file_0_view_0, column: Initial Inventory).
- d_i: Demand for product i (from table_id: file_0_view_0, column: Demand).

Decision Variables:
- x_i: Number of units of product i to fulfill, for all i ∈ I. (Domain: integer, 0 ≤ x_i ≤ min{s_i, d_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i \cdot x_i
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
   x_i \geq 0 \text{ and integer} \quad \forall i \in I
   \]

Data Mapping:
- Index set I: All rows in table_id: file_0_view_0 where Product Name starts with 'S700_'.
- r_i: Revenue (table_id: file_0_view_0, column: Revenue)
- s_i: Initial Inventory (table_id: file_0_view_0, column: Initial Inventory)
- d_i: Demand (table_id: file_0_view_0, column: Demand)