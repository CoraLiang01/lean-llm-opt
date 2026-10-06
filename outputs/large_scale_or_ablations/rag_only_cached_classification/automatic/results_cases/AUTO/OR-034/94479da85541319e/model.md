ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of baked goods (indexed by i), corresponding to all unique values in Frenchbakerydailysales.csv, column "Product Name".

Parameters:
- 𝑟𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Revenue per unit of baked good i (Frenchbakerydailysales.csv, "Revenue").
- 𝑖𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦ᵢ: Initial inventory available for baked good i (Frenchbakerydailysales.csv, "Initial Inventory").
- 𝑑𝑒𝑚𝑎𝑛𝑑ᵢ: Demand for baked good i (Frenchbakerydailysales.csv, "Demand").

Decision Variables:
- 𝑥ᵢ: Quantity of baked good i to fulfill (continuous, 0 ≤ 𝑥ᵢ ≤ min{𝑖𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦ᵢ, 𝑑𝑒𝑚𝑎𝑛𝑑ᵢ}).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} 𝑟𝑒𝑣𝑒𝑛𝑢𝑒ᵢ \cdot 𝑥ᵢ
  \]

Constraints:
1. Inventory constraint:
   \[
   𝑥ᵢ \leq 𝑖𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦ᵢ \quad \forall i \in 𝑰
   \]
2. Demand constraint:
   \[
   𝑥ᵢ \leq 𝑑𝑒𝑚𝑎𝑛𝑑ᵢ \quad \forall i \in 𝑰
   \]
3. Non-negativity:
   \[
   𝑥ᵢ \geq 0 \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: Frenchbakerydailysales.csv, "Product Name"
- Parameter 𝑟𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Frenchbakerydailysales.csv, "Revenue"
- Parameter 𝑖𝑛𝑣𝑒𝑛𝑡𝑜𝑟𝑦ᵢ: Frenchbakerydailysales.csv, "Initial Inventory"
- Parameter 𝑑𝑒𝑚𝑎𝑛𝑑ᵢ: Frenchbakerydailysales.csv, "Demand"

All data is sourced from table_id: file_0_view_0.