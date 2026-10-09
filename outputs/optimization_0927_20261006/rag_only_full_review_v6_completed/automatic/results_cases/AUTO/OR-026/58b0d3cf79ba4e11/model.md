Abstract Mathematical Optimization Model

Index Sets:
- 𝑰: Set of all products classified as 'Fashion' (from table_id: file_0_view_0, column: Product Name).

Parameters:
- r_i: Revenue per unit of product i ∈ 𝑰 (file_0_view_0, column: Revenue).
- d_i: Demand quantity for product i ∈ 𝑰 (file_0_view_0, column: Demand).
- s_i: Initial inventory for product i ∈ 𝑰 (file_0_view_0, column: Initial Inventory).

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill (integer, 0 ≤ x_i ≤ min{d_i, s_i}).

Objective:
Maximize total revenue from fulfilled demand:
\[
\max \sum_{i \in 𝑰} r_i \cdot x_i
\]

Constraints:
1. Inventory constraint for each product:
\[
x_i \leq s_i \quad \forall i \in 𝑰
\]
2. Demand fulfillment constraint for each product:
\[
x_i \leq d_i \quad \forall i \in 𝑰
\]
3. Non-negativity and integrality:
\[
x_i \geq 0 \text{ and integer} \quad \forall i \in 𝑰
\]

Data Mapping:
- Index set 𝑰, and parameters r_i, d_i, s_i are sourced from table_id: file_0_view_0 (SupermarketSales.csv), columns: Product Name, Revenue, Demand, Initial Inventory, with the filter: Product Name starts with 'Fashion accessories_' (i.e., products classified as 'Fashion').