Mathematical Model (Abstract Formulation)

Index Sets:
- 𝑰: Set of areas (indexed by i), corresponding to all ProductName values in file_1_view_0.

Parameters:
- 𝑣ᵢ: Benefit coefficient for area i. (file_1_view_0, column: Value)
- 𝑤ᵢ: Development unit weight for area i. (file_1_view_0, column: Weight)
- C: Overall development capacity. (file_0_view_0, column: Capacity)

Decision Variables:
- xᵢ ∈ ℤ₊: Integer number of development units to allocate daily in area i, ∀i ∈ 𝑰.

Objective:
Maximize total benefit:
  max ∑_{i ∈ 𝑰} vᵢ xᵢ

Subject to:
1. Overall capacity constraint:
  ∑_{i ∈ 𝑰} wᵢ xᵢ ≤ C

2. Nonnegativity and integrality:
  xᵢ ∈ {0, 1, 2, ...}  ∀i ∈ 𝑰

Data Mapping

Index Sets:
- 𝑰: All rows in file_1_view_0, indexed by ProductName.

Parameters:
- vᵢ: file_1_view_0, column Value, for area i (ProductName).
- wᵢ: file_1_view_0, column Weight, for area i (ProductName).
- C: file_0_view_0, column Capacity (single value).

Decision Variables:
- xᵢ: Integer, for each i = ProductName in file_1_view_0.

Objective:
- Maximize ∑_{i ∈ 𝑰} (file_1_view_0:Value)[i] × xᵢ

Constraint:
- ∑_{i ∈ 𝑰} (file_1_view_0:Weight)[i] × xᵢ ≤ (file_0_view_0:Capacity)

Variable domains:
- xᵢ ∈ ℤ₊  ∀i = ProductName in file_1_view_0