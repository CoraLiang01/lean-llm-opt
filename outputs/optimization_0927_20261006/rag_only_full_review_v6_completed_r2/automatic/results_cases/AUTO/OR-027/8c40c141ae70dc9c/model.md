Abstract Optimization Model

Index Sets:
- Let I be the set of products classified under ‘Organ’, indexed by i. (From Sub Category values with prefix "Organic" in table_id: file_0_view_0.)

Parameters:
- Revenue_i: Revenue per unit of product i. (From column Revenue, table_id: file_0_view_0.)
- InitialInventory_i: Initial inventory available for product i. (From column Initial Inventory, table_id: file_0_view_0.)
- Demand_i: Deterministic demand for product i. (From column Demand, table_id: file_0_view_0.)

Decision Variables:
- x_i: Number of units of product i to fulfill, integer, with 0 ≤ x_i ≤ min(InitialInventory_i, Demand_i), ∀i ∈ I.

Objective:
Maximize total revenue:
\[
\max \sum_{i \in I} Revenue_i \cdot x_i
\]

Constraints:
1. Inventory constraint:
\[
x_i \leq InitialInventory_i, \quad \forall i \in I
\]
2. Demand constraint:
\[
x_i \leq Demand_i, \quad \forall i \in I
\]
3. Non-negativity and integrality:
\[
x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
\]

Data Mapping:
- Index set I: All records in table_id: file_0_view_0, where Sub Category has prefix "Organic".
- Revenue_i: Revenue column, table_id: file_0_view_0.
- InitialInventory_i: Initial Inventory column, table_id: file_0_view_0.
- Demand_i: Demand column, table_id: file_0_view_0.

No additional constraints or selection rules are imposed beyond those specified in the user query and the validated filter.