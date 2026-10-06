ABSTRACT MATHEMATICAL MODEL

Sets:
- 𝑰: Set of areas (indexed by i), corresponding to all ProductName values in file_1_view_0.

Parameters:
- 𝑣ᵢ: Development benefit per unit scale in area i. (file_1_view_0, column: Value)
- 𝑤ᵢ: Development resource required per unit scale in area i. (file_1_view_0, column: Weight)
- 𝐶: Total available development capacity. (file_0_view_0, column: Capacity)

Decision Variables:
- 𝑥ᵢ ≥ 0 and integer: Scale of development per day in area i.

Objective:
Maximize total development benefit:
\[
\max \sum_{i \in 𝑰} v_i x_i
\]

Subject to:
- Total development resource usage does not exceed capacity:
\[
\sum_{i \in 𝑰} w_i x_i \leq C
\]
- Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in 𝑰
\]

DATA MAPPING

- Set 𝑰: All rows in file_1_view_0, indexed by ProductName.
- Parameter 𝑣ᵢ: file_1_view_0, column Value, for each ProductName.
- Parameter 𝑤ᵢ: file_1_view_0, column Weight, for each ProductName.
- Parameter 𝐶: file_0_view_0, column Capacity (single value).
- Decision variable 𝑥ᵢ: Scale of development per day in area i (ProductName), integer and ≥ 0.