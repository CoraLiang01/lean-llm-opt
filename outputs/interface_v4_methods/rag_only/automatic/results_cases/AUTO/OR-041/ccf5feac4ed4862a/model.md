ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑰: Set of real estate areas (indexed by i), corresponding to ProductName in products.csv.

Parameters:
- 𝑏𝑒𝑛𝑒𝑓𝑖𝑡ᵢ: Development benefit per unit scale in area i. (products.csv: Value)
- 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ: Resource consumption per unit scale in area i. (products.csv: Weight)
- 𝐶: Overall development capacity. (capacity.csv: Capacity)

Decision Variables:
- 𝑥ᵢ: Scale of development per day in area i (nonnegative integer), ∀i∈𝑰.

Objective:
Maximize total development benefit:
\[
\max \sum_{i \in \mathcal{I}} benefit_i \cdot x_i
\]

Subject to:
- Overall development capacity constraint:
\[
\sum_{i \in \mathcal{I}} weight_i \cdot x_i \leq C
\]
- Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
\]

DATA MAPPING

Index Set:
- 𝑰: All rows in products.csv, using ProductName as the area identifier (table_id: file_1_view_0, column: ProductName).

Parameters:
- 𝑏𝑒𝑛𝑒𝑓𝑖𝑡ᵢ: file_1_view_0, column: Value, for each ProductName.
- 𝑤𝑒𝑖𝑔ℎ𝑡ᵢ: file_1_view_0, column: Weight, for each ProductName.
- 𝐶: file_0_view_0, column: Capacity (single value).

Decision Variables:
- 𝑥ᵢ: For each ProductName in file_1_view_0.

All data is used in original row order, with no synthesized or omitted identifiers.