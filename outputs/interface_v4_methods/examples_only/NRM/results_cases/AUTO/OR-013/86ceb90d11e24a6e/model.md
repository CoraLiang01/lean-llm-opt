Abstract Mathematical Optimization Model

Index Sets:
- 𝑰: Set of all “4U” products (indexed by i), where each i corresponds to a unique [Product Name] in table_id file_0_view_0.

Parameters:
- 𝑟ᵢ: Revenue per unit for product i ([Revenue], table_id file_0_view_0)
- 𝑑ᵢ: Demand for product i ([Demand], table_id file_0_view_0)
- 𝑠ᵢ: Initial inventory for product i ([Initial Inventory], table_id file_0_view_0)

Decision Variables:
- 𝑥ᵢ: Number of units of product i to fulfill (integer, 0 ≤ 𝑥ᵢ ≤ min{𝑑ᵢ, 𝑠ᵢ})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} r_i x_i
  \]

Constraints:
1. Inventory and demand fulfillment limits:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in 𝑰
   \]
   (Equivalently, two constraints per product:)
   \[
   x_i \leq d_i \quad \forall i \in 𝑰
   \]
   \[
   x_i \leq s_i \quad \forall i \in 𝑰
   \]
2. Integer fulfillment:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: All rows in table_id file_0_view_0 where [Product Name] starts with "4U"
- Parameter 𝑟ᵢ: [Revenue] column, table_id file_0_view_0
- Parameter 𝑑ᵢ: [Demand] column, table_id file_0_view_0
- Parameter 𝑠ᵢ: [Initial Inventory] column, table_id file_0_view_0

No data values or record counts are included; all references are symbolic and mapped to the exact table and columns as specified.