ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑃: Set of vehicle types (indexed by i), corresponding to ProductName in products.csv.

Parameters:
- v_i: Benefit coefficient of vehicle type i. (Data: products.csv, column "Value")
- w_i: Inventory weight per unit of vehicle type i. (Data: products.csv, column "Weight")
- C: Total inventory capacity. (Data: capacity.csv, column "Capacity")

Decision Variables:
- x_i: Number of units of vehicle type i to order daily. (Domain: integer, x_i ≥ 0, ∀i ∈ 𝑃)

Objective:
Maximize total benefit:
\[
\max \sum_{i \in 𝑃} v_i x_i
\]

Constraints:
1. Inventory capacity constraint:
\[
\sum_{i \in 𝑃} w_i x_i \leq C
\]
2. Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in 𝑃
\]

Data Mapping:

- 𝑃 (vehicle types): products.csv, column "ProductName", table_id: file_1_view_0
- v_i: products.csv, column "Value", table_id: file_1_view_0, key: ProductName
- w_i: products.csv, column "Weight", table_id: file_1_view_0, key: ProductName
- C: capacity.csv, column "Capacity", table_id: file_0_view_0

Each x_i is indexed by the ProductName from products.csv. All parameters are mapped directly from the supplied files as described.