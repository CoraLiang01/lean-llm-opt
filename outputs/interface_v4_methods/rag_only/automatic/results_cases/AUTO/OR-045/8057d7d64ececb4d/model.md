ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑃: Set of produce types, indexed by i. (From file_1_view_0: ProductName)

Parameters:
- v_i: Benefit per unit of produce i. (Data Mapping: file_1_view_0, column "Value", key "ProductName")
- w_i: Weight per unit of produce i. (Data Mapping: file_1_view_0, column "Weight", key "ProductName")
- C: Total inventory capacity. (Data Mapping: file_0_view_0, column "Capacity")

Decision Variables:
- x_i: Number of units of produce i to order daily. (Integer, x_i ≥ 0, ∀i ∈ 𝑃)

Objective:
Maximize total benefit:
\[
\max \sum_{i \in 𝑃} v_i x_i
\]

Subject to:
- Inventory capacity constraint:
\[
\sum_{i \in 𝑃} w_i x_i \leq C
\]
- Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in 𝑃
\]

DATA MAPPING

- Index set 𝑃 and keys: file_1_view_0, column "ProductName"
- Parameter v_i: file_1_view_0, column "Value", key "ProductName"
- Parameter w_i: file_1_view_0, column "Weight", key "ProductName"
- Parameter C: file_0_view_0, column "Capacity"

All data is used as returned, with no omitted or synthesized values.