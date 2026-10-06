ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑰: Set of areas (indexed by i), corresponding to each ProductName in products.csv.

Parameters:
- 𝑏ᵢ: Benefit coefficient for area i. (from products.csv, column Value)
- C: Overall development capacity. (from capacity.csv, column Capacity)

Decision Variables:
- xᵢ: Integer, xᵢ ≥ 0. Daily scale of development in area i.

Objective:
Maximize total benefit:
  max ∑_{i ∈ 𝑰} 𝑏ᵢ xᵢ

Constraints:
1. Capacity constraint:
  ∑_{i ∈ 𝑰} xᵢ ≤ C

2. Integrality and nonnegativity:
  xᵢ ∈ ℤ₊ ∀ i ∈ 𝑰

DATA MAPPING

- Index set 𝑰: All rows in products.csv (table_id: file_1_view_0), using ProductName as area identifier.
- Parameter 𝑏ᵢ: products.csv (file_1_view_0), column Value, mapped by ProductName.
- Parameter C: capacity.csv (file_0_view_0), column Capacity, single scalar.
- Decision variable xᵢ: Defined for each ProductName in products.csv (file_1_view_0).

No other columns are used; Weight is ignored as not referenced in the user description.