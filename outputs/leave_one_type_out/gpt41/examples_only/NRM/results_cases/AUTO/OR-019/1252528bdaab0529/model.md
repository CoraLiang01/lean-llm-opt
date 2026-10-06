ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products classified as ‘27in’ (from SalesDataAnalysis.csv, Product Name column, filtered by prefix '27in').

Parameters:
- r_i: Revenue per unit of product i ∈ I (SalesDataAnalysis.csv, Revenue column).
- d_i: Demand quantity for product i ∈ I (SalesDataAnalysis.csv, Demand column).
- s_i: Initial inventory for product i ∈ I (SalesDataAnalysis.csv, Initial Inventory column).

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, x_i ≥ 0).

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
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

Data Mapping:
- Index set I, and parameters r_i, d_i, s_i are sourced from SalesDataAnalysis.csv (table_id: file_0_view_0), using:
    - Product Name (filtered by prefix '27in') → index set I
    - Revenue → r_i
    - Demand → d_i
    - Initial Inventory → s_i