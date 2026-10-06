ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑃: Set of vehicle types (ProductName from products.csv, table_id: file_1_view_0)

Parameters:
- v_p: Profit per unit of vehicle type p ∈ 𝑃 (Value from products.csv, table_id: file_1_view_0, column: Value)
- w_p: Inventory weight per unit of vehicle type p ∈ 𝑃 (Weight from products.csv, table_id: file_1_view_0, column: Weight)
- C: Overall inventory capacity (Capacity from capacity.csv, table_id: file_0_view_0, column: Capacity)

Decision Variables:
- x_p: Number of vehicles of type p ∈ 𝑃 to order per day (integer, x_p ≥ 0)

Objective:
Maximize total profit:
\[
\max \sum_{p \in 𝑃} v_p \cdot x_p
\]

Subject to:
- Inventory capacity constraint:
\[
\sum_{p \in 𝑃} w_p \cdot x_p \leq C
\]
- Nonnegativity and integrality:
\[
x_p \in \mathbb{Z}_+, \quad \forall p \in 𝑃
\]

Data Mapping:
- 𝑃: All ProductName values from products.csv (table_id: file_1_view_0, column: ProductName)
- v_p: Value from products.csv (table_id: file_1_view_0, column: Value), mapped by ProductName
- w_p: Weight from products.csv (table_id: file_1_view_0, column: Weight), mapped by ProductName
- C: Capacity from capacity.csv (table_id: file_0_view_0, column: Capacity)

Each x_p is indexed by the original ProductName; do not synthesize or reorder IDs. All parameters are mapped directly from the supplied files as described.