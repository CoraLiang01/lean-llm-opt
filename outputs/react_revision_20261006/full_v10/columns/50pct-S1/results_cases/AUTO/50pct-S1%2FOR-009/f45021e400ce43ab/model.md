Mathematical Model

Index Sets:
- Let 𝑰 be the set of areas available for development, with each area identified by ProductName in file_1_view_0.

Parameters:
- For each area i ∈ 𝑰:
    - v_i: development benefit of area i (Value from file_1_view_0, column Value)
    - w_i: development resource required per unit scale in area i (Weight from file_1_view_0, column Weight)
- C: overall development capacity (Capacity from file_0_view_0, column Capacity)

Decision Variables:
- For each area i ∈ 𝑰:
    - x_i ≥ 0, integer: scale of development per day in area i

Objective:
Maximize total development benefit:
$$
\max \sum_{i \in 𝑰} v_i x_i
$$

Subject to:
Overall development capacity constraint:
$$
\sum_{i \in 𝑰} w_i x_i \leq C
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in 𝑰
$$

Data Mapping

Index Sets:
- 𝑰: All ProductName values in file_1_view_0 (products.csv), column ProductName

Parameters:
- v_i: file_1_view_0 (products.csv), column Value, for each ProductName
- w_i: file_1_view_0 (products.csv), column Weight, for each ProductName
- C: file_0_view_0 (capacity.csv), column Capacity

Variables:
- x_i: scale of development per day in area i, for each i ∈ 𝑰

Objective:
- Maximize total benefit: sum over i ∈ 𝑰 of v_i x_i

Constraint:
- Total resource use: sum over i ∈ 𝑰 of w_i x_i ≤ C

Variable domains:
- x_i ∈ ℤ₊ for all i ∈ 𝑰