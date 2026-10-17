ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of all products classified under ‘Fashion’. Each product i ∈ 𝑰 corresponds to a row where "Product Name" starts with "Fashion" in SupermarketSales.csv.

Parameters:
- r_i: Revenue per unit of product i. [Source: SupermarketSales.csv, column "Revenue", table_id: file_0_view_0]
- s_i: Initial inventory of product i. [Source: SupermarketSales.csv, column "Initial Inventory", table_id: file_0_view_0]
- d_i: Demand for product i (deterministic, known). [Source: SupermarketSales.csv, column "Demand", table_id: file_0_view_0]

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, 0 ≤ x_i ≤ min{s_i, d_i}), ∀i ∈ 𝑰.

Objective:
- Maximize total revenue from fulfilled Fashion products:
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
   x_i \geq 0,\quad x_i \in \mathbb{Z} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰 and all parameters (r_i, s_i, d_i) are sourced from SupermarketSales.csv, table_id: file_0_view_0, restricted to rows where "Product Name" starts with "Fashion" (CSVQA-applied filter: {"column": "Product Name", "operator": "prefix", "value": "Fashion"}).
- r_i: "Revenue" column
- s_i: "Initial Inventory" column
- d_i: "Demand" column

No additional constraints or selection rules are imposed beyond those stated in the user query and the CSVQA-applied filter.