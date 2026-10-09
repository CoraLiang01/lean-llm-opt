Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- 𝑰: Set of all products classified under ‘Books’ (from Product_Name in file_0_view_0).

Parameters:
- r_i: Revenue per unit of product i ∈ 𝑰 (from Revenue, file_0_view_0).
- d_i: Demand quantity for product i ∈ 𝑰 (from Demand, file_0_view_0).
- s_i: Initial inventory for product i ∈ 𝑰 (from Initial Inventory, file_0_view_0).

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill (continuous, x_i ≥ 0).

Objective:
Maximize total revenue from fulfilled ‘Books’ products:
\[
\max \sum_{i \in 𝑰} r_i \cdot x_i
\]

Constraints:
1. Demand fulfillment cannot exceed demand:
\[
x_i \leq d_i \quad \forall i \in 𝑰
\]
2. Inventory constraint:
\[
x_i \leq s_i \quad \forall i \in 𝑰
\]
3. Non-negativity:
\[
x_i \geq 0 \quad \forall i \in 𝑰
\]

Data Mapping:
- Index set 𝑰: All records in file_0_view_0 where Product_Name starts with "Books".
- Parameter r_i: Revenue column in file_0_view_0, mapped by Product_Name.
- Parameter d_i: Demand column in file_0_view_0, mapped by Product_Name.
- Parameter s_i: Initial Inventory column in file_0_view_0, mapped by Product_Name.
- Decision variable x_i: Number of units fulfilled for each i ∈ 𝑰.

All data is sourced from table_id file_0_view_0, columns: Product_Name, Revenue, Demand, Initial Inventory.