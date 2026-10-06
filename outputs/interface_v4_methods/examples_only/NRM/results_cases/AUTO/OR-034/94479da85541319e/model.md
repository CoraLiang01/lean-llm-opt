ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let 𝑰 be the set of all baked goods, indexed by i. (From Frenchbakerydailysales.csv, column: Product Name)

Parameters:
- 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Revenue per unit of baked good i. (Frenchbakerydailysales.csv, column: Revenue)
- 𝐼𝑛𝑖𝑡𝐼𝑛𝑣ᵢ: Initial inventory available for baked good i. (Frenchbakerydailysales.csv, column: Initial Inventory)
- 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: Deterministic demand for baked good i. (Frenchbakerydailysales.csv, column: Demand)

Decision Variables:
- 𝑥ᵢ ≥ 0: Quantity of baked good i to fulfill (continuous or integer, as appropriate).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ \cdot 𝑥ᵢ
  \]

Constraints:
1. Cannot fulfill more than demand:
   \[
   𝑥ᵢ \leq 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ \quad \forall i \in 𝑰
   \]
2. Cannot fulfill more than initial inventory:
   \[
   𝑥ᵢ \leq 𝐼𝑛𝑖𝑡𝐼𝑛𝑣ᵢ \quad \forall i \in 𝑰
   \]
3. Non-negativity:
   \[
   𝑥ᵢ \geq 0 \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: Frenchbakerydailysales.csv, table_id: file_0_view_0, column: Product Name
- Parameter 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Frenchbakerydailysales.csv, table_id: file_0_view_0, column: Revenue
- Parameter 𝐼𝑛𝑖𝑡𝐼𝑛𝑣ᵢ: Frenchbakerydailysales.csv, table_id: file_0_view_0, column: Initial Inventory
- Parameter 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: Frenchbakerydailysales.csv, table_id: file_0_view_0, column: Demand