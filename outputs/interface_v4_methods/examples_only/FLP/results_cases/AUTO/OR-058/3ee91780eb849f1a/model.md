Mathematical Model for Supplier Activation and Product Sourcing

Sets:
- Let \( I = \{1,2,3,4,5,6\} \) denote the set of suppliers, corresponding to S1–S6.
- Let \( J = \{1,2,3,4,5,6\} \) denote the set of stores, corresponding to C1–C6.

Parameters:
- Fixed cost for opening supplier \( i \):
  \[
  f_i = 
  \begin{cases}
    98.88 & \text{if } i=1 \ (\text{S1}) \\
    99.73 & \text{if } i=2 \ (\text{S2}) \\
    94.01 & \text{if } i=3 \ (\text{S3}) \\
    93.77 & \text{if } i=4 \ (\text{S4}) \\
    107.59 & \text{if } i=5 \ (\text{S5}) \\
    112.65 & \text{if } i=6 \ (\text{S6}) \\
  \end{cases}
  \]

- Demand at each store \( j \):
  \[
  d_j = 
  \begin{cases}
    216 & \text{if } j=1 \ (\text{C1}) \\
    216 & \text{if } j=2 \ (\text{C2}) \\
    216 & \text{if } j=3 \ (\text{C3}) \\
    144 & \text{if } j=4 \ (\text{C4}) \\
    144 & \text{if } j=5 \ (\text{C5}) \\
    144 & \text{if } j=6 \ (\text{C6}) \\
  \end{cases}
  \]

- Transportation cost per unit from supplier \( i \) to store \( j \) (\( c_{ij} \)):
  \[
  C = 
  \begin{bmatrix}
  0.08 & 52.33 & 73.57 & 1237.33 & 0.07 & 112.16 \\
  46.02 & 175.23 & 2026.83 & 299.89 & 966.53 & 1590.42 \\
  1031.74 & 78.13 & 99.02 & 277.07 & 884.45 & 1800.86 \\
  868.75 & 94.2 & 1776.34 & 285.48 & 868.85 & 86.55 \\
  1577 & 760.15 & 2090.19 & 43.2 & 1577.12 & 1095.17 \\
  49.14 & 4.33 & 2079.57 & 277.04 & 1032.01 & 1543.49 \\
  \end{bmatrix}
  \]
  where row \( i \) corresponds to supplier S\(i\), and column \( j \) to store C\(j\).

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if supplier \( i \) is operational (open), 0 otherwise.
- \( x_{ij} \geq 0 \): Quantity of Adidas products supplied from supplier \( i \) to store \( j \).

Objective Function:
Minimize the total cost, which is the sum of fixed costs for open suppliers and the total transportation cost:
\[
\min \sum_{i=1}^{6} f_i y_i + \sum_{i=1}^{6} \sum_{j=1}^{6} c_{ij} x_{ij}
\]
where:
- \( f_i \) is the fixed cost for supplier \( i \)
- \( c_{ij} \) is the transportation cost per unit from supplier \( i \) to store \( j \)
- \( y_i \) is the binary variable for supplier \( i \)
- \( x_{ij} \) is the quantity shipped from supplier \( i \) to store \( j \)

Constraints:
1. Demand satisfaction at each store:
   \[
   \sum_{i=1}^{6} x_{ij} = d_j \quad \forall j = 1,\ldots,6
   \]
   (Each store’s demand must be fully met.)

2. Supplier activation constraint:
   \[
   x_{ij} \leq d_j y_i \quad \forall i = 1,\ldots,6;\ j = 1,\ldots,6
   \]
   (A supplier can only supply to a store if it is open.)

3. Variable domains:
   \[
   y_i \in \{0,1\} \quad \forall i = 1,\ldots,6
   \]
   \[
   x_{ij} \geq 0 \quad \forall i = 1,\ldots,6;\ j = 1,\ldots,6
   \]

Summary of Parameters (explicit vectors/matrices):

- Fixed cost vector:
  \[
  f = [98.88,\ 99.73,\ 94.01,\ 93.77,\ 107.59,\ 112.65]
  \]

- Demand vector:
  \[
  d = [216,\ 216,\ 216,\ 144,\ 144,\ 144]
  \]

- Transportation cost matrix:
  \[
  C = 
  \begin{bmatrix}
  0.08 & 52.33 & 73.57 & 1237.33 & 0.07 & 112.16 \\
  46.02 & 175.23 & 2026.83 & 299.89 & 966.53 & 1590.42 \\
  1031.74 & 78.13 & 99.02 & 277.07 & 884.45 & 1800.86 \\
  868.75 & 94.2 & 1776.34 & 285.48 & 868.85 & 86.55 \\
  1577 & 760.15 & 2090.19 & 43.2 & 1577.12 & 1095.17 \\
  49.14 & 4.33 & 2079.57 & 277.04 & 1032.01 & 1543.49 \\
  \end{bmatrix}
  \]

This model determines which suppliers to open and how much each should supply to each store to minimize the total cost while meeting all store demands.