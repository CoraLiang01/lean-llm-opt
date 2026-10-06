ABSTRACT MATHEMATICAL MODEL

Sets:
- 𝑃: Set of vehicle types (indexed by i), corresponding to all ProductName values in products.csv.

Parameters:
- 𝑣𝑎𝑙𝑢𝑒ᵢ: Profit from selling one unit of vehicle type i. (from products.csv, Value)
- 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ: Inventory space consumed by one unit of vehicle type i. (from products.csv, Weight)
- 𝐶: Total available inventory capacity. (from capacity.csv, Capacity)

Decision Variables:
- 𝑥ᵢ: Number of vehicles of type i to order per day (integer, 𝑥ᵢ ≥ 0, ∀i ∈ 𝑃).

Objective:
- Maximize total profit:
  
  maximize ∑_{i∈𝑃} 𝑣𝑎𝑙𝑢𝑒ᵢ · 𝑥ᵢ

Constraints:
1. Inventory capacity constraint:
  
  ∑_{i∈𝑃} 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ · 𝑥ᵢ ≤ 𝐶

2. Nonnegativity and integrality:
  
  𝑥ᵢ ∈ {0, 1, 2, ...} ∀i ∈ 𝑃

DATA MAPPING

- Set 𝑃: All rows in file_1_view_0 (products.csv), indexed by ProductName.
- Parameter 𝑣𝑎𝑙𝑢𝑒ᵢ: file_1_view_0, column Value, for each ProductName.
- Parameter 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ: file_1_view_0, column Weight, for each ProductName.
- Parameter 𝐶: file_0_view_0, column Capacity (single value).
- Decision variable 𝑥ᵢ: Number of vehicles of type i to order per day, indexed by ProductName from file_1_view_0.