ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products i classified as ‘27in’ (from table_id: file_0_view_0, column: Product Name).

Parameters:
- r_i: Revenue per unit of product i (from table_id: file_0_view_0, column: Revenue).
- d_i: Demand quantity for product i (from table_id: file_0_view_0, column: Demand).
- s_i: Initial inventory for product i (from table_id: file_0_view_0, column: Initial Inventory).

Decision Variables:
- x_i: Number of units of product i to fulfill, ∀i ∈ I. (Domain: integer, 0 ≤ x_i ≤ min{d_i, s_i})

Objective:
- Maximize total revenue from fulfilled units:
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
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I, parameters r_i, d_i, s_i are sourced from table_id: file_0_view_0 in SalesDataAnalysis.csv:
    - Product Name (prefix ‘27in’): defines I
    - Revenue: r_i
    - Demand: d_i
    - Initial Inventory: s_i

No data or constraints have been omitted or invented. The model is abstract and symbolic, as requested.