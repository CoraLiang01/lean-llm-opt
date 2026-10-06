ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products classified as ‘27in’ (from SalesDataAnalysis.csv, rows where [Product Name] starts with '27in').

Parameters:
- r_i: Revenue per unit of product i ∈ I (from SalesDataAnalysis.csv, [Revenue], table_id: file_0_view_0)
- d_i: Demand quantity for product i ∈ I (from SalesDataAnalysis.csv, [Demand], table_id: file_0_view_0)
- s_i: Initial inventory for product i ∈ I (from SalesDataAnalysis.csv, [Initial Inventory], table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min{d_i, s_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i \cdot x_i
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
   x_i \geq 0,\ x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I: All rows in SalesDataAnalysis.csv (table_id: file_0_view_0) where [Product Name] starts with '27in'
- r_i: [Revenue] column, table_id: file_0_view_0
- d_i: [Demand] column, table_id: file_0_view_0
- s_i: [Initial Inventory] column, table_id: file_0_view_0