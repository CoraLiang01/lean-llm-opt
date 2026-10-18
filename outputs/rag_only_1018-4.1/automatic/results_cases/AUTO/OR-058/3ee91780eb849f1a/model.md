Let:
- Suppliers: \( I = \{S1, S2, S3, S4, S5, S6\} \)
- Stores: \( J = \{C1, C2, C3, C4, C5, C6\} \)

Parameters:
- Fixed costs for each supplier:
  - \( f_{S1} = 98.88 \)
  - \( f_{S2} = 99.73 \)
  - \( f_{S3} = 94.01 \)
  - \( f_{S4} = 93.77 \)
  - \( f_{S5} = 107.59 \)
  - \( f_{S6} = 112.65 \)
- Demand at each store:
  - \( d_{C1} = 216 \)
  - \( d_{C2} = 216 \)
  - \( d_{C3} = 216 \)
  - \( d_{C4} = 144 \)
  - \( d_{C5} = 144 \)
  - \( d_{C6} = 144 \)
- Transportation costs per unit (\( c_{ij} \)) from supplier \( i \) to store \( j \):

\[
\begin{array}{c|cccccc}
 & C1 & C2 & C3 & C4 & C5 & C6 \\
\hline
S1 & 0.08 & 52.33 & 73.57 & 1237.33 & 0.07 & 112.16 \\
S2 & 46.02 & 175.23 & 2026.83 & 299.89 & 966.53 & 1590.42 \\
S3 & 1031.74 & 78.13 & 99.02 & 277.07 & 884.45 & 1800.86 \\
S4 & 868.75 & 94.2 & 1776.34 & 285.48 & 868.85 & 86.55 \\
S5 & 1577 & 760.15 & 2090.19 & 43.2 & 1577.12 & 1095.17 \\
S6 & 49.14 & 4.33 & 2079.57 & 277.04 & 1032.01 & 1543.49 \\
\end{array}
\]

Decision variables:
- \( y_i \in \{0,1\} \) for each supplier \( i \in I \): 1 if supplier \( i \) is operational, 0 otherwise.
- \( x_{ij} \geq 0 \) for each supplier \( i \in I \), store \( j \in J \): quantity supplied from \( i \) to \( j \).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{subject to} \quad
& \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& x_{ij} \leq d_j y_i \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I, j \in J \\
\end{align*}
\]

Where:
- \( f_i \) is the fixed cost for supplier \( i \)
- \( c_{ij} \) is the transportation cost per unit from supplier \( i \) to store \( j \)
- \( d_j \) is the demand at store \( j \)

All parameters (vectors and matrices):

- \( I = [S1, S2, S3, S4, S5, S6] \)
- \( J = [C1, C2, C3, C4, C5, C6] \)
- \( f = [98.88, 99.73, 94.01, 93.77, 107.59, 112.65] \)
- \( d = [216, 216, 216, 144, 144, 144] \)
- \( C = \)
\[
\begin{bmatrix}
0.08 & 52.33 & 73.57 & 1237.33 & 0.07 & 112.16 \\
46.02 & 175.23 & 2026.83 & 299.89 & 966.53 & 1590.42 \\
1031.74 & 78.13 & 99.02 & 277.07 & 884.45 & 1800.86 \\
868.75 & 94.2 & 1776.34 & 285.48 & 868.85 & 86.55 \\
1577 & 760.15 & 2090.19 & 43.2 & 1577.12 & 1095.17 \\
49.14 & 4.33 & 2079.57 & 277.04 & 1032.01 & 1543.49 \\
\end{bmatrix}
\]

This model determines which suppliers to activate and how much each should supply to each store to minimize total cost while meeting all store demands.