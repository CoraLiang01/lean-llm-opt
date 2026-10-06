ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑃: Set of drug products, indexed by p (ProductName from products.csv)

Parameters:
- 𝑏ₚ: Benefit per unit of drug product p (Value column, file_1_view_0)
- 𝑤ₚ: Stock weight per unit of drug product p (Weight column, file_1_view_0)
- 𝐶: Overall stock capacity for the pharmacy chain (Capacity column, file_0_view_0)

Decision Variables:
- 𝑥ₚ: Number of units of drug product p to order each day (integer, 𝑥ₚ ≥ 0)

Objective:
Maximize total benefit:
\[
\max \sum_{p \in P} b_p x_p
\]

Constraints:
1. Stock capacity constraint:
\[
\sum_{p \in P} w_p x_p \leq C
\]
2. Nonnegativity and integrality:
\[
x_p \in \mathbb{Z}_+, \quad \forall p \in P
\]

DATA MAPPING

- 𝑃: All rows in file_1_view_0 (products.csv), ProductName
- 𝑏ₚ: file_1_view_0, Value (for each ProductName)
- 𝑤ₚ: file_1_view_0, Weight (for each ProductName)
- 𝐶: file_0_view_0, Capacity (single value)
- 𝑥ₚ: Decision variable for each ProductName in file_1_view_0

This model selects integer order quantities for each drug to maximize total benefit, subject to the overall stock capacity.