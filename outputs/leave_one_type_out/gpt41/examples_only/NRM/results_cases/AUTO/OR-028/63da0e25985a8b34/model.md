Abstract Optimization Model

Index Sets:
- \( I \): Set of products (indexed by \( i \))

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from column 'Revenue')
- \( d_i \): Demand for product \( i \) (from column 'Demand')
- \( s_i \): Initial inventory for product \( i \) (from column 'Initial Inventory')

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Table: file_0_view_0 (from WomenClothingEcommerceSalesData.csv)
    - Index set \( I \): 'Product Name'
    - Parameter \( r_i \): 'Revenue'
    - Parameter \( d_i \): 'Demand'
    - Parameter \( s_i \): 'Initial Inventory'