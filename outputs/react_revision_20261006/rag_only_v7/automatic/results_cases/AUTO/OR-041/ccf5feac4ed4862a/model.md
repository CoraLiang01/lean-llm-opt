Mathematical Optimization Model

Index Sets:
- 𝑰: Set of areas available for development (ProductName in file_1_view_0)

Parameters:
- 𝑣ᵢ: Development benefit per unit scale in area i (Value, file_1_view_0)
- 𝑤ᵢ: Development capacity consumed per unit scale in area i (Weight, file_1_view_0)
- C: Overall development capacity (Capacity, file_0_view_0)

Decision Variables:
- xᵢ ≥ 0, integer: Scale of development per day in area i, ∀i ∈ 𝑰

Objective:
Maximize total development benefit:
  max ∑_{i∈𝑰} vᵢ xᵢ

Subject to:
- Overall development capacity constraint:
  ∑_{i∈𝑰} wᵢ xᵢ ≤ C
- Nonnegativity and integrality:
  xᵢ ∈ ℤ₊ ∀i ∈ 𝑰

Data Mapping

Index Sets:
- 𝑰: All ProductName values in file_1_view_0 (preserve row order)

Parameters:
- vᵢ: file_1_view_0.Value, indexed by file_1_view_0.ProductName
- wᵢ: file_1_view_0.Weight, indexed by file_1_view_0.ProductName
- C: file_0_view_0.Capacity

Decision Variables:
- xᵢ: Scale of development per day in area i, indexed by file_1_view_0.ProductName

Constraints:
- Capacity: sum over i of (file_1_view_0.Weight × xᵢ) ≤ file_0_view_0.Capacity

Objective:
- Maximize sum over i of (file_1_view_0.Value × xᵢ)