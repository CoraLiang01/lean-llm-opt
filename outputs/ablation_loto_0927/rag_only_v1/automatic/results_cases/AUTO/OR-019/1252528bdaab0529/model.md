ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products classified as ‘27in’ (from SalesDataAnalysis.csv, Product Name with prefix ‘27in’).

Parameters:
- r_i: Revenue per unit of product i ∈ I (SalesDataAnalysis.csv, Revenue, table_id: file_0_view_0)
- d_i: Demand for product i ∈ I (SalesDataAnalysis.csv, Demand, table_id: file_0_view_0)
- s_i: Initial inventory for product i ∈ I (SalesDataAnalysis.csv, Initial Inventory, table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min{d_i, s_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i \cdot x_i
  \]

Constraints:
1. Demand fulfillment constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I, and parameters r_i, d_i, s_i are sourced from SalesDataAnalysis.csv (table_id: file_0_view_0), using rows where Product Name has prefix ‘27in’. Specifically:
    - Product Name → index set I
    - Revenue → r_i
    - Demand → d_i
    - Initial Inventory → s_i