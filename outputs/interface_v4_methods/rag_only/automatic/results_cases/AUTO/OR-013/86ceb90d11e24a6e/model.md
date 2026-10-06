Abstract Mathematical Optimization Model

Index Sets:
- I: Set of all “4U” products (indexed by i ∈ I)

Parameters:
- r_i: Revenue per unit for product i (from Revenue column)
- d_i: Demand for product i during the sales horizon (from Demand column)
- s_i: Initial inventory available for product i (from Initial Inventory column)

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, x_i ≥ 0)

Objective:
- Maximize total revenue from fulfilled quantities:
  \[
  \max \sum_{i \in I} r_i \cdot x_i
  \]

Constraints:
1. Inventory and demand fulfillment limits:
   \[
   0 \leq x_i \leq \min\{s_i, d_i\} \quad \forall i \in I
   \]
   (x_i must not exceed either available inventory or realized demand)

2. Integer constraints:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

Data Mapping:
- Index set I: All rows in OnlineSalesinUSA.csv where "Product Name" starts with "4U" (table_id: file_0_view_0, column: Product Name)
- Parameter r_i: OnlineSalesinUSA.csv, column "Revenue", table_id: file_0_view_0
- Parameter d_i: OnlineSalesinUSA.csv, column "Demand", table_id: file_0_view_0
- Parameter s_i: OnlineSalesinUSA.csv, column "Initial Inventory", table_id: file_0_view_0

No data values or literal record counts are included; all sets and parameters are defined symbolically and mapped to their exact sources.