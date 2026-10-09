Mathematical Optimization Model

Index Sets:
- 𝑃: Set of produce types (ProductName in file_1_view_0)

Parameters:
- 𝑣_p: Value (benefit) per unit of produce p ∈ 𝑃 (Value column in file_1_view_0)
- 𝑤_p: Weight per unit of produce p ∈ 𝑃 (Weight column in file_1_view_0)
- C: Total inventory capacity (Capacity column in file_0_view_0)

Decision Variables:
- x_p: Number of units of produce p ∈ 𝑃 to order daily (integer, x_p ≥ 0)

Objective:
Maximize total benefit:
\[
\max \sum_{p \in 𝑃} v_p \cdot x_p
\]

Subject to:
1. Inventory capacity constraint:
\[
\sum_{p \in 𝑃} w_p \cdot x_p \leq C
\]
2. Nonnegativity and integrality:
\[
x_p \in \mathbb{Z}_+, \quad \forall p \in 𝑃
\]

Data Mapping

Index Sets:
- 𝑃: All ProductName in file_1_view_0

Parameters:
- v_p: file_1_view_0, column Value, indexed by ProductName
- w_p: file_1_view_0, column Weight, indexed by ProductName
- C: file_0_view_0, column Capacity

Decision Variables:
- x_p: Integer, ≥0, indexed by ProductName

Constraints:
- Inventory capacity: sum over ProductName of (Weight × x_p) ≤ Capacity

Objective:
- Maximize sum over ProductName of (Value × x_p)