Mathematical Model (Abstract Formulation)

Index Sets:
- 𝑃: Set of drug types (indexed by i), corresponding to ProductName in file_1_view_0.

Parameters:
- 𝑣ᵢ: Benefit coefficient of drug type i (Value column in file_1_view_0).
- 𝑤ᵢ: Weight per unit of drug type i (Weight column in file_1_view_0).
- 𝐶: Overall inventory capacity (Capacity column in file_0_view_0).

Decision Variables:
- 𝑥ᵢ: Number of units of drug type i to order daily (integer, 𝑥ᵢ ≥ 0).

Objective:
Maximize total benefit:
\[
\max \sum_{i \in P} v_i x_i
\]

Subject to:
- Inventory capacity constraint:
\[
\sum_{i \in P} w_i x_i \leq C
\]
- Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in P
\]

Data Mapping:
- Index set 𝑃: All rows in file_1_view_0, identified by ProductName.
- Parameter 𝑣ᵢ: file_1_view_0, column Value, for each ProductName.
- Parameter 𝑤ᵢ: file_1_view_0, column Weight, for each ProductName.
- Parameter 𝐶: file_0_view_0, column Capacity (single value).
- Decision variable 𝑥ᵢ: Number of units to order for each ProductName in file_1_view_0.