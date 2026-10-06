ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products classified as ‘27in’ (from Salesorders.csv, Product Name with prefix '27in').

Parameters:
- r_i: Revenue per unit of product i ∈ I (Salesorders.csv, Revenue, table_id: file_0_view_0, column: Revenue).
- s_i: Initial inventory of product i ∈ I (Salesorders.csv, Initial Inventory, table_id: file_0_view_0, column: Initial Inventory).
- d_i: Demand for product i ∈ I (Salesorders.csv, Demand, table_id: file_0_view_0, column: Demand).

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min{s_i, d_i}).

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
- Index set I, and parameters r_i, s_i, d_i are sourced from Salesorders.csv (table_id: file_0_view_0), using:
    - Product Name (prefix '27in') for I,
    - Revenue for r_i,
    - Initial Inventory for s_i,
    - Demand for d_i.