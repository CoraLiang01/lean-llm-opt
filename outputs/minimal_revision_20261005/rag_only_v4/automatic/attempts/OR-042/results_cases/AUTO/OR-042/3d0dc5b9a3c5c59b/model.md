ABSTRACT MATHEMATICAL MODEL

Sets:
- 𝑰: Set of drug types, indexed by i. (Each i corresponds to a ProductName in file_1_view_0.)

Parameters:
- 𝑣ᵢ: Benefit coefficient of drug type i. (file_1_view_0, column Value)
- 𝑤ᵢ: Weight per unit of drug type i. (file_1_view_0, column Weight)
- 𝐶: Total inventory capacity. (file_0_view_0, column Capacity)

Decision Variables:
- 𝑥ᵢ: Number of units of drug type i to order daily. (Integer, 𝑥ᵢ ≥ 0)

Objective:
Maximize total benefit:
\[
\max \sum_{i \in \mathcal{I}} v_i x_i
\]

Subject to:
- Inventory capacity constraint:
\[
\sum_{i \in \mathcal{I}} w_i x_i \leq C
\]
- Integer and nonnegativity constraints:
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
\]

DATA MAPPING

- 𝑰: All rows in file_1_view_0, indexed by ProductName.
- 𝑣ᵢ: file_1_view_0, column Value, for each ProductName.
- 𝑤ᵢ: file_1_view_0, column Weight, for each ProductName.
- 𝐶: file_0_view_0, column Capacity, row 0.
- 𝑥ᵢ: Decision variable for each ProductName in file_1_view_0.