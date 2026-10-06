ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑰: Set of vehicle types (indexed by i), corresponding to all ProductName values in products.csv.

Parameters:
- 𝑣𝑎𝑙𝑢𝑒ᵢ: Benefit coefficient of vehicle type i. (from products.csv, Value)
- 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ: Inventory weight per unit of vehicle type i. (from products.csv, Weight)
- 𝐶: Total inventory capacity. (from capacity.csv, Capacity)

Decision Variables:
- 𝑥ᵢ: Number of units of vehicle type i to order daily. Integer, 𝑥ᵢ ≥ 0, ∀i∈𝑰.

Objective:
- Maximize total benefit:
  
  maximize ∑_{i∈𝑰} 𝑣𝑎𝑙𝑢𝑒ᵢ · 𝑥ᵢ

Constraint:
- Inventory capacity:
  
  ∑_{i∈𝑰} 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ · 𝑥ᵢ ≤ 𝐶

- Integer and nonnegativity:
  
  𝑥ᵢ ∈ ℤ₊, ∀i∈𝑰

DATA MAPPING

- Index set 𝑰: All ProductName in table_id file_1_view_0, column ProductName.
- Parameter 𝑣𝑎𝑙𝑢𝑒ᵢ: file_1_view_0, columns ProductName (for i), Value (as numeric).
- Parameter 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ: file_1_view_0, columns ProductName (for i), Weight (as numeric).
- Parameter 𝐶: file_0_view_0, column Capacity (as numeric, single value).
- Decision variable 𝑥ᵢ: defined for each i ∈ ProductName in file_1_view_0.