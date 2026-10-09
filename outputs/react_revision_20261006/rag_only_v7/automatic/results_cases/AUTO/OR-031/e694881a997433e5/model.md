Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- 𝑰: Set of dairy products, indexed by i (from Full_Product_Name in file_0_view_0)

Parameters:
- Revenue_i: Revenue per unit of product i (from Revenue, file_0_view_0)
- Demand_i: Demand quantity for product i (from Demand, file_0_view_0)
- Inventory_i: Initial inventory available for product i (from Initial Inventory, file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, 0 ≤ x_i ≤ min(Demand_i, Inventory_i))

Objective:
Maximize total revenue:
\[
\max \sum_{i \in \mathcal{I}} \text{Revenue}_i \cdot x_i
\]

Constraints:
1. Demand fulfillment constraint:
\[
x_i \leq \text{Demand}_i \quad \forall i \in \mathcal{I}
\]
2. Inventory limit constraint:
\[
x_i \leq \text{Inventory}_i \quad \forall i \in \mathcal{I}
\]
3. Non-negativity and integrality:
\[
x_i \geq 0,\quad x_i \in \mathbb{Z} \quad \forall i \in \mathcal{I}
\]

Data Mapping:
- Index set 𝑰: All Full_Product_Name values in table_id file_0_view_0, column Full_Product_Name
- Parameter Revenue_i: file_0_view_0, column Revenue
- Parameter Demand_i: file_0_view_0, column Demand
- Parameter Inventory_i: file_0_view_0, column Initial Inventory
- Decision variable x_i: Number of units fulfilled for each i ∈ 𝑰

All parameters are mapped directly from the specified columns in table_id file_0_view_0.