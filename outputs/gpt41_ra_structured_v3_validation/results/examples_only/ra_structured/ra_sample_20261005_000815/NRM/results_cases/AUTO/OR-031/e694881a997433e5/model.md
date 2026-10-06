ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑃: Set of dairy products, indexed by i.

Parameters:
- 𝑑𝑒𝑚𝑎𝑛𝑑ᵢ: Demand for product i. (Source: DairyGoodsSalesDataset.csv, column Demand, table_id: file_0_view_0)
- 𝑖𝑛𝑣ᵢ: Initial inventory for product i. (Source: DairyGoodsSalesDataset.csv, column Initial Inventory, table_id: file_0_view_0)
- 𝑟ᵢ: Revenue per unit for product i. (Source: DairyGoodsSalesDataset.csv, column Revenue, table_id: file_0_view_0)

Decision Variables:
- 𝑥ᵢ ≥ 0: Number of units of product i to fulfill (continuous or integer, as appropriate).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} r_i x_i
  \]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq inv_i \quad \forall i \in P
   \]
2. Demand constraint:
   \[
   x_i \leq demand_i \quad \forall i \in P
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in P
   \]

Data Mapping:
- Set of products (P): DairyGoodsSalesDataset.csv, column Full_Product_Name, table_id: file_0_view_0
- Initial inventory (inv_i): DairyGoodsSalesDataset.csv, column Initial Inventory, table_id: file_0_view_0
- Demand (demand_i): DairyGoodsSalesDataset.csv, column Demand, table_id: file_0_view_0
- Revenue per unit (r_i): DairyGoodsSalesDataset.csv, column Revenue, table_id: file_0_view_0