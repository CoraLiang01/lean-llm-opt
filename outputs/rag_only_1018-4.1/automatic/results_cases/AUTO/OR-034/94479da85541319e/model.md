ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of all baked goods, indexed by i (corresponds to "Product Name" in table_id: file_0_view_0).

Parameters:
- 𝑟ᵢ: Revenue per unit of baked good i (from "Revenue", table_id: file_0_view_0).
- 𝑑ᵢ: Demand for baked good i (from "Demand", table_id: file_0_view_0).
- 𝑠ᵢ: Initial inventory available for baked good i (from "Initial Inventory", table_id: file_0_view_0).

Decision Variables:
- 𝑥ᵢ ≥ 0: Quantity of baked good i to fulfill (continuous or integer, as appropriate).

Objective:
Maximize total revenue:
\[
\max \sum_{i \in \mathcal{I}} r_i x_i
\]

Constraints:
1. Inventory constraint for each baked good:
\[
x_i \leq s_i \quad \forall i \in \mathcal{I}
\]
2. Demand fulfillment constraint for each baked good:
\[
x_i \leq d_i \quad \forall i \in \mathcal{I}
\]
3. Non-negativity:
\[
x_i \geq 0 \quad \forall i \in \mathcal{I}
\]

Data Mapping:
- Index set 𝑰 and all parameters (𝑟ᵢ, 𝑑ᵢ, 𝑠ᵢ) are sourced from table_id: file_0_view_0, columns:
    - "Product Name" → 𝑰 (baked good index)
    - "Revenue" → 𝑟ᵢ
    - "Demand" → 𝑑ᵢ
    - "Initial Inventory" → 𝑠ᵢ

No additional filters or eligibility rules were applied; all records in file_0_view_0 are included as per the query.