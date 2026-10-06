ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑰: Set of drug types (indexed by i), corresponding to ProductName in products.csv.

Parameters:
- 𝑣𝑎𝑙𝑢𝑒ᵢ: Benefit coefficient of drug type i. (from products.csv, column Value)
- 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ: Per-unit weight of drug type i. (from products.csv, column Weight)
- 𝐶: Overall inventory capacity. (from capacity.csv, column Capacity)

Decision Variables:
- 𝑥ᵢ: Number of units of drug type i to order daily. (integer, 𝑥ᵢ ≥ 0, ∀i ∈ 𝑰)

Objective:
Maximize total benefit:
\[
\max \sum_{i \in 𝑰} 𝑣𝑎𝑙𝑢𝑒ᵢ \cdot 𝑥ᵢ
\]

Subject to:
- Inventory capacity constraint:
\[
\sum_{i \in 𝑰} 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ \cdot 𝑥ᵢ \leq 𝐶
\]
- Integer and nonnegativity constraints:
\[
𝑥ᵢ \in \mathbb{Z}_+, \quad \forall i \in 𝑰
\]

Data Mapping:

- Index set 𝑰: All rows in products.csv, with business ID ProductName (table_id: file_1_view_0, column: ProductName)
- Parameter 𝑣𝑎𝑙𝑢𝑒ᵢ: products.csv, column Value (table_id: file_1_view_0, column: Value), keyed by ProductName
- Parameter 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ: products.csv, column Weight (table_id: file_1_view_0, column: Weight), keyed by ProductName
- Parameter 𝐶: capacity.csv, column Capacity (table_id: file_0_view_0, column: Capacity), single value

All variables and parameters are aligned by the explicit ProductName business ID. No data is omitted or synthesized.