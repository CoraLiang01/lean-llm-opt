ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑃: Set of bread types (ProductName from products.csv, table_id: file_1_view_0)

Parameters:
- v_p: Expected profit per unit of bread type p ∈ 𝑃 (Value from file_1_view_0, column: Value)
- w_p: Storage weight per unit of bread type p ∈ 𝑃 (Weight from file_1_view_0, column: Weight)
- C: Total storage capacity (Capacity from file_0_view_0, column: Capacity)

Decision Variables:
- x_p: Number of units of bread type p ∈ 𝑃 to order each day (integer, x_p ≥ 0)

Objective:
Maximize total expected profit:
\[
\max \sum_{p \in 𝑃} v_p \cdot x_p
\]

Subject to:
- Storage capacity constraint:
\[
\sum_{p \in 𝑃} w_p \cdot x_p \leq C
\]
- Nonnegativity and integrality:
\[
x_p \in \mathbb{Z}_{\geq 0} \quad \forall p \in 𝑃
\]

DATA MAPPING

- 𝑃: All rows in file_1_view_0 (products.csv), ProductName
- v_p: file_1_view_0, column Value, indexed by ProductName
- w_p: file_1_view_0, column Weight, indexed by ProductName
- C: file_0_view_0, column Capacity
- x_p: Decision variable for each ProductName in file_1_view_0

All data is used as provided; no business IDs are synthesized.