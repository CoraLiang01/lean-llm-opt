ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑰: Set of vehicle types (ProductName from products.csv)

Parameters:
- 𝑣𝑎𝑙𝑢𝑒ᵢ: Benefit coefficient of vehicle type i ∈ 𝑰 (Value from products.csv, table_id: file_1_view_0, column: Value)
- 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ: Inventory weight/unit for vehicle type i ∈ 𝑰 (Weight from products.csv, table_id: file_1_view_0, column: Weight)
- 𝐶: Total inventory capacity (Capacity from capacity.csv, table_id: file_0_view_0, column: Capacity)

Decision Variables:
- 𝑥ᵢ: Number of units of vehicle type i ∈ 𝑰 to order daily (integer, 𝑥ᵢ ≥ 0)

Objective:
Maximize total benefit:
\[
\max \sum_{i \in 𝑰} 𝑣𝑎𝑙𝑢𝑒ᵢ \cdot 𝑥ᵢ
\]

Subject to:
Inventory capacity constraint:
\[
\sum_{i \in 𝑰} 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ \cdot 𝑥ᵢ \leq 𝐶
\]

Integrality and nonnegativity:
\[
𝑥ᵢ \in \mathbb{Z}_+, \quad \forall i \in 𝑰
\]

DATA MAPPING

- 𝑰: All ProductName values from products.csv (table_id: file_1_view_0, column: ProductName)
- 𝑣𝑎𝑙𝑢𝑒ᵢ: Value from products.csv (table_id: file_1_view_0, column: Value), for each i = ProductName
- 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ: Weight from products.csv (table_id: file_1_view_0, column: Weight), for each i = ProductName
- 𝐶: Capacity from capacity.csv (table_id: file_0_view_0, column: Capacity), single value

All vehicle types and their coefficients are indexed by ProductName. The model maximizes total benefit from ordered vehicles, subject to the total inventory capacity, with integer order quantities.