ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of products, indexed by i.

Parameters:
- Revenue_i: Revenue per unit of product i. [Source: SalesDatainBusinesses.csv, column 'Revenue']
- Demand_i: Demand quantity for product i. [Source: SalesDatainBusinesses.csv, column 'Demand']
- Inventory_i: Initial inventory available for product i. [Source: SalesDatainBusinesses.csv, column 'Initial Inventory']

Decision Variables:
- x_i: Number of units of product i to fulfill. (Domain: integer, 0 ≤ x_i ≤ min{Demand_i, Inventory_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq Inventory_i \quad \forall i \in 𝑰
   \]
2. Demand constraint for each product:
   \[
   x_i \leq Demand_i \quad \forall i \in 𝑰
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: All unique 'Product Name' values from SalesDatainBusinesses.csv (table_id: file_0_view_0, column: 'Product Name')
- Revenue_i: SalesDatainBusinesses.csv (table_id: file_0_view_0, column: 'Revenue')
- Demand_i: SalesDatainBusinesses.csv (table_id: file_0_view_0, column: 'Demand')
- Inventory_i: SalesDatainBusinesses.csv (table_id: file_0_view_0, column: 'Initial Inventory')
- Decision variable x_i: defined for each i ∈ 𝑰

This model maximizes total revenue by optimally fulfilling product demand, subject to inventory and demand limits for each product.