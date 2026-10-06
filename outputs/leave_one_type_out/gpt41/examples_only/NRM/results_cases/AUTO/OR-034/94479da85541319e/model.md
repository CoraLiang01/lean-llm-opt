Abstract Optimization Model

Index Sets:
- 𝑖 ∈ 𝐼: Set of baked goods (from Product Name column)

Parameters:
- 𝑟𝑒𝑣_𝑖: Revenue per unit of baked good 𝑖 (from Revenue column)
- 𝑖𝑛𝑣_𝑖: Initial inventory available for baked good 𝑖 (from Initial Inventory column)
- 𝑑𝑒𝑚_𝑖: Demand for baked good 𝑖 (from Demand column)

Decision Variables:
- 𝑥_𝑖 ≥ 0: Quantity of baked good 𝑖 to fulfill (continuous or integer, as appropriate)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} rev_i \cdot x_i
  \]

Constraints:
1. Cannot fulfill more than demand:
   \[
   x_i \leq dem_i \quad \forall i \in I
   \]
2. Cannot fulfill more than initial inventory:
   \[
   x_i \leq inv_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in I
   \]

Data Mapping:
- Table: file_0_view_0 (Frenchbakerydailysales.csv)
  - Index set 𝐼: Product Name
  - Parameter 𝑟𝑒𝑣_𝑖: Revenue
  - Parameter 𝑖𝑛𝑣_𝑖: Initial Inventory
  - Parameter 𝑑𝑒𝑚_𝑖: Demand