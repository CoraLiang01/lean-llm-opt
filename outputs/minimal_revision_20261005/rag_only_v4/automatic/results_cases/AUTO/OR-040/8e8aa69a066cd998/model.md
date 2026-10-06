ABSTRACT MATHEMATICAL MODEL

Sets:
- 𝑰: Set of areas (indexed by i), corresponding to all ProductName values in products.csv.

Parameters:
- 𝑣ᵢ: Benefit coefficient for area i. (from products.csv, column Value)
- 𝑤ᵢ: Development units required per unit scale in area i. (from products.csv, column Weight)
- 𝐶: Overall development capacity. (from capacity.csv, column Capacity)

Decision Variables:
- 𝑥ᵢ: Integer, number of development units to allocate daily in area i; 𝑥ᵢ ≥ 0 and integer ∀ i ∈ 𝑰.

Objective:
- Maximize total benefit:
  maximize ∑_{i∈𝑰} 𝑣ᵢ · 𝑥ᵢ

Constraints:
1. Overall development capacity:
  ∑_{i∈𝑰} 𝑤ᵢ · 𝑥ᵢ ≤ 𝐶
2. Nonnegativity and integrality:
  𝑥ᵢ ∈ {0, 1, 2, ...} ∀ i ∈ 𝑰

DATA MAPPING

Index Set:
- 𝑰: All rows in file_1_view_0 (products.csv), identified by ProductName.

Parameters:
- 𝑣ᵢ: file_1_view_0, column Value, for ProductName = i
- 𝑤ᵢ: file_1_view_0, column Weight, for ProductName = i
- 𝐶: file_0_view_0, column Capacity, row 0

Decision Variables:
- 𝑥ᵢ: Integer variable for each ProductName in file_1_view_0

All mappings use the exact table_id and column names as above.