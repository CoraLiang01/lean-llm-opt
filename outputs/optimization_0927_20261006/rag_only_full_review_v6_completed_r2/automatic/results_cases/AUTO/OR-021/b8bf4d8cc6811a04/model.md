Abstract Mathematical Optimization Model

Index Sets:
- Let 𝑃 be the set of all clothing products, indexed by 𝑖.

Parameters:
- 𝑟𝑒𝑣_𝑖: Revenue per unit for product 𝑖. (from column 'Revenue')
- 𝑑𝑒𝑚_𝑖: Deterministic demand for product 𝑖. (from column 'Demand')
- 𝑖𝑛𝑣_𝑖: Initial inventory available for product 𝑖. (from column 'Initial Inventory')

Decision Variables:
- 𝑥_𝑖: Number of units of product 𝑖 to fulfill (integer, 𝑥_𝑖 ≥ 0).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} rev_i \cdot x_i
  \]

Constraints:
1. Inventory and demand fulfillment bounds for each product:
   \[
   0 \leq x_i \leq \min\{inv_i, dem_i\} \quad \forall i \in P
   \]
   (Equivalently, two constraints per product:)
   \[
   x_i \leq inv_i \quad \forall i \in P
   \]
   \[
   x_i \leq dem_i \quad \forall i \in P
   \]
2. Integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in P
   \]

Data Mapping:
- All data is sourced from table_id: file_0_view_0, columns:
  - Product Name: index set P
  - Revenue: parameter rev_i
  - Demand: parameter dem_i
  - Initial Inventory: parameter inv_i

No additional constraints or eligibility rules are imposed beyond those specified above. All products in the returned records are included in the index set P.