ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let 𝑰 be the set of all baked goods, indexed by i.

Parameters:
- 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Revenue per unit of baked good i. [from column 'Revenue']
- 𝐼𝐼ᵢ: Initial inventory of baked good i. [from column 'Initial Inventory']
- 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: Demand for baked good i. [from column 'Demand']

Decision Variables:
- 𝑥ᵢ ≥ 0: Quantity of baked good i to fulfill (continuous or integer, as appropriate).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ \cdot 𝑥ᵢ
  \]

Constraints:
1. Inventory constraint for each baked good:
   \[
   𝑥ᵢ \leq 𝐼𝐼ᵢ \quad \forall i \in 𝑰
   \]
2. Demand constraint for each baked good:
   \[
   𝑥ᵢ \leq 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ \quad \forall i \in 𝑰
   \]
3. Nonnegativity:
   \[
   𝑥ᵢ \geq 0 \quad \forall i \in 𝑰
   \]

Data Mapping:
- Table: Frenchbakerydailysales.csv (table_id: file_0_view_0)
  - Index set 𝑰: Product Name
  - Parameter 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: column 'Revenue'
  - Parameter 𝐼𝐼ᵢ: column 'Initial Inventory'
  - Parameter 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: column 'Demand'