ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let 𝑰 be the set of all products classified as ‘Baby’ (from table_id: file_0_view_0, column: Product Name).

Parameters:
- 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Revenue per unit of product i ∈ 𝑰 (from table_id: file_0_view_0, column: Revenue).
- 𝐼𝑛𝑖𝑡𝐼𝑛𝑣ᵢ: Initial inventory of product i ∈ 𝑰 (from table_id: file_0_view_0, column: Initial Inventory).
- 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: Demand quantity for product i ∈ 𝑰 (from table_id: file_0_view_0, column: Demand).

Decision Variables:
- 𝑥ᵢ: Number of units of product i ∈ 𝑰 to fulfill (continuous or integer, as appropriate; domain: 0 ≤ 𝑥ᵢ ≤ min{𝐼𝑛𝑖𝑡𝐼𝑛𝑣ᵢ, 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ}).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ \cdot 𝑥ᵢ
  \]

Constraints:
1. Inventory constraint:
   \[
   𝑥ᵢ \leq 𝐼𝑛𝑖𝑡𝐼𝑛𝑣ᵢ \quad \forall i \in 𝑰
   \]
2. Demand constraint:
   \[
   𝑥ᵢ \leq 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ \quad \forall i \in 𝑰
   \]
3. Non-negativity:
   \[
   𝑥ᵢ \geq 0 \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: All rows in table_id: file_0_view_0 where Product Name has prefix 'Baby'
- 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: file_0_view_0, column: Revenue
- 𝐼𝑛𝑖𝑡𝐼𝑛𝑣ᵢ: file_0_view_0, column: Initial Inventory
- 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: file_0_view_0, column: Demand