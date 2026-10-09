Abstract Optimization Model

Index Sets:
- Let 𝑰 be the set of all products with [Product Name] starting with 'Aalop' (from table_id: file_0_view_0).

Parameters (for each i ∈ 𝑰):
- r_i: Revenue per unit of product i ([Revenue], file_0_view_0)
- d_i: Demand for product i ([Demand], file_0_view_0)
- s_i: Initial inventory of product i ([Initial Inventory], file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, 0 ≤ x_i ≤ min{d_i, s_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} r_i \cdot x_i
  \]

Constraints:
1. Demand and inventory fulfillment bounds:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in 𝑰
   \]
   (Alternatively, as two constraints per product:)
   \[
   x_i \leq d_i \quad \forall i \in 𝑰
   \]
   \[
   x_i \leq s_i \quad \forall i \in 𝑰
   \]
2. Integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: All records in table_id file_0_view_0 where [Product Name] has prefix 'Aalop'.
- r_i: [Revenue] column, table_id file_0_view_0.
- d_i: [Demand] column, table_id file_0_view_0.
- s_i: [Initial Inventory] column, table_id file_0_view_0.

Note: The model uses only the records returned by the validated filter ([Product Name] prefix 'Aalop') from file_0_view_0. No restocking or in-transit inventory is permitted. Demand is deterministic and known.