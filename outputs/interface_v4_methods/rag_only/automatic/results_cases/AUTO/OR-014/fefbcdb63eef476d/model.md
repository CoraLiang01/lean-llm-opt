ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑃: Set of pizza types (indexed by i)

Parameters:
- r_i: Revenue per unit of pizza type i (source: table_id="file_0_view_0", column="Revenue")
- d_i: Demand for pizza type i (source: table_id="file_0_view_0", column="Demand")
- s_i: Initial inventory for pizza type i (source: table_id="file_0_view_0", column="Initial Inventory")

Decision Variables:
- x_i: Number of units of pizza type i to fulfill (integer, x_i ≥ 0)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} r_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each pizza type:
   \[
   x_i \leq s_i \quad \forall i \in P
   \]
2. Demand constraint for each pizza type:
   \[
   x_i \leq d_i \quad \forall i \in P
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in P
   \]

Data Mapping:
- Index set 𝑃 and all parameters (r_i, d_i, s_i) are sourced from table_id="file_0_view_0" (PizzaSalesDataset.csv):
    - Pizza type: column "Product Name"
    - Revenue: column "Revenue"
    - Demand: column "Demand"
    - Initial Inventory: column "Initial Inventory"