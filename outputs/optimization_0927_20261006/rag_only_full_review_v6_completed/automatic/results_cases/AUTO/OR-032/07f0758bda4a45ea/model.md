ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of products classified under ‘Books’ (from DifferentStoreSales.csv, rows where Product_Name starts with "Books")

Parameters:
- Revenue_i: Revenue per unit for product i ∈ 𝑰 (DifferentStoreSales.csv, column Revenue, table_id: file_0_view_0)
- InitialInventory_i: Initial inventory available for product i ∈ 𝑰 (DifferentStoreSales.csv, column Initial Inventory, table_id: file_0_view_0)
- Demand_i: Deterministic demand for product i ∈ 𝑰 (DifferentStoreSales.csv, column Demand, table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill (continuous, x_i ≥ 0)

Objective:
Maximize total revenue from fulfilled units:
\[
\max \sum_{i \in 𝑰} Revenue_i \cdot x_i
\]

Constraints:
1. Inventory constraint for each product:
\[
x_i \leq InitialInventory_i \quad \forall i \in 𝑰
\]
2. Demand fulfillment constraint for each product:
\[
x_i \leq Demand_i \quad \forall i \in 𝑰
\]
3. Non-negativity:
\[
x_i \geq 0 \quad \forall i \in 𝑰
\]

Data Mapping:
- Index set 𝑰 and all parameters (Revenue_i, InitialInventory_i, Demand_i) are sourced from DifferentStoreSales.csv, table_id: file_0_view_0, restricted to records where Product_Name starts with "Books" (CSVQA-applied filter: prefix "Books" on Product_Name).
- Revenue_i: column Revenue
- InitialInventory_i: column Initial Inventory
- Demand_i: column Demand

No additional constraints or requirements are imposed beyond those specified in the user query and the validated data subset.