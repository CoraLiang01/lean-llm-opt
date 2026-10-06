Abstract Optimization Model

Index Sets:
- 𝑰: Set of products (indexed by i)

Parameters:
- r_i: Revenue per unit of product i (from SalesDatainBusinesses.csv, column 'Revenue')
- d_i: Demand for product i (from SalesDatainBusinesses.csv, column 'Demand')
- s_i: Initial inventory available for product i (from SalesDatainBusinesses.csv, column 'Initial Inventory')

Decision Variables:
- x_i: Number of units of product i to fulfill (continuous or integer, as appropriate; 0 ≤ x_i ≤ min{d_i, s_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in \mathcal{I}} r_i \cdot x_i
  \]

Constraints:
1. Demand fulfillment constraint:
   \[
   x_i \leq d_i \quad \forall i \in \mathcal{I}
   \]
2. Inventory availability constraint:
   \[
   x_i \leq s_i \quad \forall i \in \mathcal{I}
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in \mathcal{I}
   \]

Data Mapping:
- Index set 𝑰: All unique values in 'Product Name' from SalesDatainBusinesses.csv (table_id: file_0_view_0, column: 'Product Name')
- Parameter r_i: 'Revenue' from SalesDatainBusinesses.csv (table_id: file_0_view_0, column: 'Revenue')
- Parameter d_i: 'Demand' from SalesDatainBusinesses.csv (table_id: file_0_view_0, column: 'Demand')
- Parameter s_i: 'Initial Inventory' from SalesDatainBusinesses.csv (table_id: file_0_view_0, column: 'Initial Inventory')