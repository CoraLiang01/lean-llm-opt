ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑰: Set of drug types, indexed by i. (Each i corresponds to a unique ProductName from products.csv.)

Parameters:
- 𝑣ᵢ: Benefit coefficient of drug type i. (Data Mapping: file_1_view_0, column "Value", key "ProductName")
- 𝑤ᵢ: Per-unit weight of drug type i. (Data Mapping: file_1_view_0, column "Weight", key "ProductName")
- C: Overall inventory capacity. (Data Mapping: file_0_view_0, column "Capacity")

Decision Variables:
- xᵢ: Number of units of drug type i to order daily. (Domain: integer, xᵢ ≥ 0, ∀i ∈ 𝑰)

Objective:
Maximize total benefit:
\[
\max \sum_{i \in 𝑰} v_i x_i
\]

Subject to:
- Inventory capacity constraint:
\[
\sum_{i \in 𝑰} w_i x_i \leq C
\]
- Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in 𝑰
\]

Data Mapping:
- 𝑰: All "ProductName" values in file_1_view_0 (products.csv).
- 𝑣ᵢ: "Value" column in file_1_view_0, keyed by "ProductName".
- 𝑤ᵢ: "Weight" column in file_1_view_0, keyed by "ProductName".
- C: "Capacity" column in file_0_view_0 (capacity.csv), single scalar value.

Each xᵢ is the integer number of units of drug type i to order daily, maximizing total benefit without exceeding the overall inventory weight capacity.