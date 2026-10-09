Abstract Optimization Model

Index Sets:
- 𝑃: Set of dairy products, indexed by i.

Parameters:
- 𝐼𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦ᵢ: Initial inventory of product i. (from DairyGoodsSalesDataset.csv, column 'Initial Inventory')
- 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: Demand for product i. (from DairyGoodsSalesDataset.csv, column 'Demand')
- 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Revenue per unit of product i. (from DairyGoodsSalesDataset.csv, column 'Revenue')

Decision Variables:
- 𝑥ᵢ: Number of units of product i to fulfill (continuous, 𝑥ᵢ ≥ 0)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} \text{Revenue}_i \cdot x_i
  \]

Constraints:
1. Inventory limit for each product:
   \[
   x_i \leq \text{Inventory}_i \quad \forall i \in P
   \]
2. Demand limit for each product:
   \[
   x_i \leq \text{Demand}_i \quad \forall i \in P
   \]
3. Nonnegativity:
   \[
   x_i \geq 0 \quad \forall i \in P
   \]

Data Mapping:
- Index set P (products): DairyGoodsSalesDataset.csv, column 'Full_Product_Name', table_id: file_0_view_0
- Parameter Inventoryᵢ: DairyGoodsSalesDataset.csv, column 'Initial Inventory', table_id: file_0_view_0
- Parameter Demandᵢ: DairyGoodsSalesDataset.csv, column 'Demand', table_id: file_0_view_0
- Parameter Revenueᵢ: DairyGoodsSalesDataset.csv, column 'Revenue', table_id: file_0_view_0