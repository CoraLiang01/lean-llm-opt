ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of products classified as ‘27in’ (from Salesorders.csv, filtered by Product Name prefix '27in').

Parameters:
- Revenue_i: Revenue per unit of product i ∈ I (Salesorders.csv, column: Revenue).
- InitialInventory_i: Initial inventory available for product i ∈ I (Salesorders.csv, column: Initial Inventory).
- Demand_i: Deterministic demand for product i ∈ I (Salesorders.csv, column: Demand).

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i}).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq InitialInventory_i \quad \forall i \in I
   \]
2. Demand fulfillment constraint for each product:
   \[
   x_i \leq Demand_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I, and all parameters (Revenue_i, InitialInventory_i, Demand_i) are sourced from Salesorders.csv (table_id: file_0_view_0), using:
    - Product Name (filtered by prefix '27in') for index set I,
    - Revenue (column: Revenue),
    - Initial Inventory (column: Initial Inventory),
    - Demand (column: Demand).