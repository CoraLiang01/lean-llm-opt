ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑃: Set of products/categories (indexed by i), where each i corresponds to a unique value in RetailSalesDataset.csv, column Product Name.

Parameters:
- r_i: Revenue per unit for product i (RetailSalesDataset.csv, column Revenue).
- d_i: Demand quantity for product i (RetailSalesDataset.csv, column Demand).
- s_i: Initial inventory available for product i (RetailSalesDataset.csv, column Initial Inventory).

Decision Variables:
- x_i: Quantity of product i to fulfill (continuous, x_i ≥ 0).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} r_i \cdot x_i
  \]

Constraints:
1. Inventory limit for each product:
   \[
   x_i \leq s_i \quad \forall i \in P
   \]
2. Demand fulfillment limit for each product:
   \[
   x_i \leq d_i \quad \forall i \in P
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in P
   \]

Data Mapping:
- Index set 𝑃, parameters r_i, d_i, s_i, and all variable definitions are mapped to RetailSalesDataset.csv (table_id: file_0_view_0):
    - Product Name → index set 𝑃
    - Revenue → r_i
    - Demand → d_i
    - Initial Inventory → s_i