ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products classified as 'Aalop' (from RestaurantSalesreport.csv, column Product Name, filtered by prefix 'Aalop').

Parameters:
- r_i: Revenue per unit of product i ∈ I (RestaurantSalesreport.csv, column Revenue).
- d_i: Demand for product i ∈ I (RestaurantSalesreport.csv, column Demand).
- s_i: Initial inventory of product i ∈ I (RestaurantSalesreport.csv, column Initial Inventory).

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min{d_i, s_i}).

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
- Index set I, and parameters r_i, d_i, s_i are sourced from RestaurantSalesreport.csv (table_id: file_0_view_0), using:
    - Product Name (filtered by prefix 'Aalop') → I
    - Revenue → r_i
    - Demand → d_i
    - Initial Inventory → s_i