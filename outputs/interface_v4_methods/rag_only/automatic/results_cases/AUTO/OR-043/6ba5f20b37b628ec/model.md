ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑃: Set of drug products, indexed by p (ProductName from products.csv)

Parameters:
- 𝑏ₚ: Benefit per unit of drug product p  
  Data Mapping: file_1_view_0, column 'Value', key 'ProductName'
- 𝑤ₚ: Stock weight per unit of drug product p  
  Data Mapping: file_1_view_0, column 'Weight', key 'ProductName'
- 𝐶: Overall stock capacity for the pharmacy chain  
  Data Mapping: file_0_view_0, column 'Capacity'

Decision Variables:
- 𝑥ₚ: Number of units of drug product p to order each day (integer, 𝑥ₚ ≥ 0)

Objective:
Maximize total benefit:
\[
\max \sum_{p \in P} b_p x_p
\]

Subject to:
- Stock capacity constraint:
\[
\sum_{p \in P} w_p x_p \leq C
\]
- Nonnegativity and integrality:
\[
x_p \in \mathbb{Z}_+, \quad \forall p \in P
\]

Data Mapping:
- 𝑃: All ProductName values in file_1_view_0 (products.csv)
- 𝑏ₚ: file_1_view_0, column 'Value', key 'ProductName'
- 𝑤ₚ: file_1_view_0, column 'Weight', key 'ProductName'
- 𝐶: file_0_view_0, column 'Capacity'

This model selects integer order quantities for each drug to maximize total benefit, subject to the overall stock capacity.