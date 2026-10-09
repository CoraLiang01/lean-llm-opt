ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑃: Set of dairy products, indexed by i.

Parameters:
- D_i: Demand for product i. (Source: DairyGoodsSalesDataset.csv, column 'Demand', table_id: file_0_view_0)
- I_i: Initial inventory for product i. (Source: DairyGoodsSalesDataset.csv, column 'Initial Inventory', table_id: file_0_view_0)
- r_i: Revenue per unit of product i. (Source: DairyGoodsSalesDataset.csv, column 'Revenue', table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill (continuous, x_i ≥ 0).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} r_i \cdot x_i
  \]

Constraints:
1. Inventory limit for each product:
   \[
   x_i \leq I_i \quad \forall i \in P
   \]
2. Demand fulfillment limit for each product:
   \[
   x_i \leq D_i \quad \forall i \in P
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in P
   \]

Data Mapping:
- Index set 𝑃: All unique values in 'Full_Product_Name' (DairyGoodsSalesDataset.csv, table_id: file_0_view_0)
- Parameter D_i: 'Demand' column (DairyGoodsSalesDataset.csv, table_id: file_0_view_0)
- Parameter I_i: 'Initial Inventory' column (DairyGoodsSalesDataset.csv, table_id: file_0_view_0)
- Parameter r_i: 'Revenue' column (DairyGoodsSalesDataset.csv, table_id: file_0_view_0)