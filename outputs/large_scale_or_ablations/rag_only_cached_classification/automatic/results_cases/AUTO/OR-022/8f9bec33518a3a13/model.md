ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products classified under ‘27in’. (i ∈ I)

Parameters:
- r_i: Revenue per unit of product i. (from column 'Revenue', table_id: file_0_view_0)
- d_i: Demand quantity for product i. (from column 'Demand', table_id: file_0_view_0)
- s_i: Initial inventory of product i. (from column 'Initial Inventory', table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill, ∀i ∈ I. (x_i ∈ ℤ₊, i.e., integer and ≥ 0)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I: All rows in Salesorders.csv (table_id: file_0_view_0) where 'Product Name' starts with '27in'.
- r_i: 'Revenue' column, table_id: file_0_view_0
- d_i: 'Demand' column, table_id: file_0_view_0
- s_i: 'Initial Inventory' column, table_id: file_0_view_0

This model maximizes total revenue from fulfilling demand for ‘27in’ products, subject to initial inventory and demand limits.