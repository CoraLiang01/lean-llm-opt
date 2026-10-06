ABSTRACT MATHEMATICAL OPTIMIZATION MODEL

Index Sets:
- I: Set of products with Product Name starting with 'S700_' (i ∈ I)

Parameters:
- r_i: Revenue per unit of product i (from Revenue column)
- d_i: Demand for product i (from Demand column)
- s_i: Initial inventory of product i (from Initial Inventory column)

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, x_i ≥ 0)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i \cdot x_i
  \]

Constraints:
1. Fulfillment cannot exceed demand:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. Fulfillment cannot exceed initial inventory:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

Data Mapping:
- Index set I and all parameters (r_i, d_i, s_i) are sourced from table_id: file_0_view_0, columns: 
  - Product Name (for I, filtered by prefix 'S700_')
  - Revenue (for r_i)
  - Demand (for d_i)
  - Initial Inventory (for s_i)