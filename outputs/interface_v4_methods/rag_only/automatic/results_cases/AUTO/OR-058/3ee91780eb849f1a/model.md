Let:
- \( I = \{1,2,3,4,5,6\} \) be the set of suppliers, corresponding to S1, S2, S3, S4, S5, S6.
- \( J = \{1,2,3,4,5,6\} \) be the set of stores, corresponding to C1, C2, C3, C4, C5, C6.

Parameters:
- Fixed cost for each supplier \( i \):
  \[
  f = 
  \begin{bmatrix}
  98.88 \\
  99.73 \\
  94.01 \\
  93.77 \\
  107.59 \\
  112.65 \\
  \end{bmatrix}
  \]
  where \( f_i \) is the fixed cost for supplier \( S_i \).

- Demand for each store \( j \):
  \[
  d = 
  \begin{bmatrix}
  216 \\
  216 \\
  216 \\
  144 \\
  144 \\
  144 \\
  \end{bmatrix}
  \]
  where \( d_j \) is the demand for store \( C_j \).

- Transportation cost per unit from supplier \( i \) to store \( j \):
  \[
  c = 
  \begin{bmatrix}
  0.08 & 52.33 & 73.57 & 1237.33 & 0.07 & 112.16 \\
  46.02 & 175.23 & 2026.83 & 299.89 & 966.53 & 1590.42 \\
  1031.74 & 78.13 & 99.02 & 277.07 & 884.45 & 1800.86 \\
  868.75 & 94.2 & 1776.34 & 285.48 & 868.85 & 86.55 \\
  1577 & 760.15 & 2090.19 & 43.2 & 1577.12 & 1095.17 \\
  49.14 & 4.33 & 2079.57 & 277.04 & 1032.01 & 1543.49 \\
  \end{bmatrix}
  \]
  where \( c_{ij} \) is the transportation cost per unit from supplier \( S_i \) to store \( C_j \).

Decision variables:
- \( y_i \in \{0,1\} \): 1 if supplier \( S_i \) is operational (open), 0 otherwise.
- \( x_{ij} \geq 0 \): quantity of goods supplied from supplier \( S_i \) to store \( C_j \).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^6 f_i y_i + \sum_{i=1}^6 \sum_{j=1}^6 c_{ij} x_{ij} \\
\text{Subject to:} \quad & \sum_{i=1}^6 x_{ij} = d_j \quad \forall j = 1,\ldots,6 \\
& x_{ij} \leq d_j y_i \quad \forall i = 1,\ldots,6;\; j = 1,\ldots,6 \\
& y_i \in \{0,1\} \quad \forall i = 1,\ldots,6 \\
& x_{ij} \geq 0 \quad \forall i = 1,\ldots,6;\; j = 1,\ldots,6 \\
\end{align*}
\]

Where:
- \( f_i \) is as listed above for each supplier.
- \( d_j \) is as listed above for each store.
- \( c_{ij} \) is as given in the transportation cost matrix above.

This model determines which suppliers to activate (minimizing fixed and transportation costs) and how much each supplier should ship to each store to meet all demands.