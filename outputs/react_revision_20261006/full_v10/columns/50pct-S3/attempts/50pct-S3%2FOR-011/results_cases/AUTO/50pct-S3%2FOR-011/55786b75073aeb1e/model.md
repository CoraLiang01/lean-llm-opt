Mathematical Model

Index Sets:
- Let 𝑰 be the set of products, with each product identified by ProductName from file_1_view_0.

Parameters:
- 𝑣ᵢ: Value of product i (file_1_view_0, column Value)
- 𝑤ᵢ: Weight of product i (file_1_view_0, column Weight)
- 𝐶: Overall stock capacity (file_0_view_0, column Capacity)

Decision Variables:
- 𝑥ᵢ ∈ ℤ₊ : Number of units of product i to order each day, ∀i ∈ 𝑰

Objective:
Maximize total benefit:
$$
\max \sum_{i \in \mathcal{I}} v_i x_i
$$

Subject to:
Overall stock capacity constraint:
$$
\sum_{i \in \mathcal{I}} w_i x_i \leq C
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{I}
$$

Data Mapping

Index Sets:
- 𝑰: All ProductName in file_1_view_0

Parameters:
- 𝑣ᵢ: file_1_view_0, column Value, keyed by ProductName
- 𝑤ᵢ: file_1_view_0, column Weight, keyed by ProductName
- 𝐶: file_0_view_0, column Capacity

Decision Variables:
- 𝑥ᵢ: Number of units to order of product i (ProductName in file_1_view_0), nonnegative integer

Objective:
- Maximize total value: sum over i of Value × x_i

Constraint:
- Total weight ordered does not exceed Capacity from file_0_view_0

All mappings use the exact column and table_id as above.