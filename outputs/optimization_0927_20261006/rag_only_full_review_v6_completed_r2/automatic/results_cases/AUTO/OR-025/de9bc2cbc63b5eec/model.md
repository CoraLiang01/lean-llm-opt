Abstract Mathematical Optimization Model

Index Sets:
- Let 𝑰 be the set of all TABLET smartphone models, indexed by i.

Parameters:
- 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Revenue per unit for model i ∈ 𝑰.
- 𝐼𝑛𝑣ᵢ: Initial Inventory for model i ∈ 𝑰.
- 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: Demand quantity for model i ∈ 𝑰.

Decision Variables:
- 𝑥ᵢ ≥ 0: Number of units of model i ∈ 𝑰 to fulfill (integer or continuous, as appropriate).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ \cdot 𝑥ᵢ
  \]

Constraints:
1. Inventory and demand fulfillment bounds for each model:
   \[
   0 \leq 𝑥ᵢ \leq \min\{𝐼𝑛𝑣ᵢ, 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ\} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰 and all parameters (𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ, 𝐼𝑛𝑣ᵢ, 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ) are sourced from SmartphoneRetailOutletSalesData.csv, table_id: file_0_view_0, using only records where the Product Name column has prefix 'TABLET'. Specifically:
    - 𝑰: All records with Product Name prefix 'TABLET'
    - 𝑅𝑒𝑣𝑒𝑛𝑢𝑒ᵢ: Revenue column
    - 𝐼𝑛𝑣ᵢ: Initial Inventory column
    - 𝐷𝑒𝑚𝑎𝑛𝑑ᵢ: Demand column

Notes:
- The model uses only the subset of products classified as 'TABLET' per the validated filter.
- All constraints and the objective are defined symbolically; no literal data values are included.