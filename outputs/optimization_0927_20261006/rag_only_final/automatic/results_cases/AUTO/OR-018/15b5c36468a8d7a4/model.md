Abstract Optimization Model

Index Sets:
- 𝑰: Set of all products i classified under ‘Baby’ (from Salesdata.csv, see Data Mapping).

Parameters:
- 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Revenue per unit of product i (Salesdata.csv, column: Revenue)
- 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: Demand for product i (Salesdata.csv, column: Demand)
- 𝐼𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦ᵢ: Initial inventory of product i (Salesdata.csv, column: Initial Inventory)

Decision Variables:
- 𝑥ᵢ: Number of units of product i to fulfill (integer, 0 ≤ 𝑥ᵢ ≤ min{𝐷𝑒𝑚𝑎𝑛𝑑ᵢ, 𝐼𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦ᵢ})

Objective:
- Maximize total revenue from ‘Baby’ products:
  \[
  \max \sum_{i \in 𝑰} 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ \cdot 𝑥ᵢ
  \]

Constraints:
1. Demand fulfillment constraint:
   \[
   0 \leq 𝑥ᵢ \leq 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ \quad \forall i \in 𝑰
   \]
2. Inventory constraint:
   \[
   0 \leq 𝑥ᵢ \leq 𝐼𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦ᵢ \quad \forall i \in 𝑰
   \]
3. Integrality:
   \[
   𝑥ᵢ \in \mathbb{Z}_{\geq 0} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: All rows in Salesdata.csv (table_id: file_0_view_0) where Product Name starts with "Baby"
- 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Salesdata.csv, column "Revenue", table_id: file_0_view_0
- 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: Salesdata.csv, column "Demand", table_id: file_0_view_0
- 𝐼𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦ᵢ: Salesdata.csv, column "Initial Inventory", table_id: file_0_view_0