ABSTRACT MATHEMATICAL MODEL

Sets:
- 𝑃: Set of vehicle types (indexed by i), corresponding to all ProductName values in products.csv.

Parameters:
- 𝑣ᵢ: Benefit coefficient of vehicle type i. (from Value column)
- 𝑤ᵢ: Inventory weight per unit of vehicle type i. (from Weight column)
- 𝐶: Total inventory capacity. (from Capacity column in capacity.csv)

Decision Variables:
- 𝑥ᵢ: Number of units of vehicle type i to order daily. Integer, 𝑥ᵢ ≥ 0, ∀ i ∈ 𝑃.

Objective:
- Maximize total benefit:
  
  maximize ∑_{i∈𝑃} 𝑣ᵢ · 𝑥ᵢ

Subject to:
- Inventory capacity constraint:
  
  ∑_{i∈𝑃} 𝑤ᵢ · 𝑥ᵢ ≤ 𝐶

- Integer and nonnegativity constraints:
  
  𝑥ᵢ ∈ ℤ₊, ∀ i ∈ 𝑃

DATA MAPPING

- Set 𝑃: All rows in file_1_view_0, indexed by ProductName.
- Parameter 𝑣ᵢ: file_1_view_0, column Value, for each ProductName.
- Parameter 𝑤ᵢ: file_1_view_0, column Weight, for each ProductName.
- Parameter 𝐶: file_0_view_0, column Capacity, single value.
- Decision variable 𝑥ᵢ: Number of units to order for each ProductName in file_1_view_0.