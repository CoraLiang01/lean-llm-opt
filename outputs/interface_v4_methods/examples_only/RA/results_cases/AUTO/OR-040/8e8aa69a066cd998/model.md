ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑰: Set of areas (indexed by i), corresponding to ProductName in products.csv.

Parameters:
- 𝑏𝑒𝑛𝑒𝑓𝑖𝑡ᵢ: Benefit coefficient for area i.  
  Data Mapping: products.csv (file_1_view_0), column "Value", indexed by "ProductName".
- 𝐶: Overall development capacity.  
  Data Mapping: capacity.csv (file_0_view_0), column "Capacity".

Decision Variables:
- 𝑥ᵢ: Integer, scale of development in area i per day (𝑥ᵢ ≥ 0, integer), ∀i ∈ 𝑰.

Objective:
- Maximize total benefit:
  \[
  \max \sum_{i \in 𝑰} 𝑏𝑒𝑛𝑒𝑓𝑖𝑡ᵢ \cdot 𝑥ᵢ
  \]

Constraints:
1. Overall capacity constraint:
   \[
   \sum_{i \in 𝑰} 𝑥ᵢ \leq 𝐶
   \]
2. Nonnegativity and integrality:
   \[
   𝑥ᵢ \in \mathbb{Z}_+, \quad \forall i \in 𝑰
   \]

DATA MAPPING

- Index set 𝑰: All rows in products.csv (file_1_view_0), column "ProductName".
- Parameter 𝑏𝑒𝑛𝑒𝑓𝑖𝑡ᵢ: products.csv (file_1_view_0), column "Value", indexed by "ProductName".
- Parameter 𝐶: capacity.csv (file_0_view_0), column "Capacity", row 0.
- Decision variable 𝑥ᵢ: One variable for each "ProductName" in products.csv (file_1_view_0).

No other columns or constraints are used. The model maximizes total benefit from integer daily development scales across all areas, subject to the single overall capacity.