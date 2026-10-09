ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑃: Set of products, indexed by i.

Parameters:
- Revenue_i: Revenue per unit for product i. (from column 'Revenue')
- Demand_i: Demand quantity for product i. (from column 'Demand')
- Inventory_i: Initial inventory available for product i. (from column 'Initial Inventory')

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, 0 ≤ x_i ≤ min{Demand_i, Inventory_i})

Objective:
Maximize total revenue from fulfilled units:
\[
\max \sum_{i \in P} Revenue_i \cdot x_i
\]

Constraints:
1. Demand fulfillment cannot exceed demand:
\[
x_i \leq Demand_i \quad \forall i \in P
\]
2. Demand fulfillment cannot exceed initial inventory:
\[
x_i \leq Inventory_i \quad \forall i \in P
\]
3. Non-negativity and integrality:
\[
x_i \geq 0 \text{ and integer} \quad \forall i \in P
\]

Data Mapping:
- Index set 𝑃 and all parameters are sourced from table_id: file_0_view_0, columns:
    - Product Name → 𝑃 (set of products)
    - Revenue → Revenue_i
    - Demand → Demand_i
    - Initial Inventory → Inventory_i
- All records in file_0_view_0 are used (no filters applied).