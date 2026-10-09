Let:
- \( I = \{S1, S2, S3, S4, S5, S6\} \) be the set of suppliers.
- \( J = \{C1, C2, C3, C4, C5, C6\} \) be the set of stores.

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
- Transportation costs per unit from supplier \( i \) to store \( j \) (\( c_{ij} \)):
  - \( c_{S1,C1} = 0.08 \), \( c_{S1,C2} = 52.33 \), \( c_{S1,C3} = 73.57 \), \( c_{S1,C4} = 1237.33 \), \( c_{S1,C5} = 0.07 \), \( c_{S1,C6} = 112.16 \)
  - \( c_{S2,C1} = 46.02 \), \( c_{S2,C2} = 175.23 \), \( c_{S2,C3} = 2026.83 \), \( c_{S2,C4} = 299.89 \), \( c_{S2,C5} = 966.53 \), \( c_{S2,C6} = 1590.42 \)
  - \( c_{S3,C1} = 1031.74 \), \( c_{S3,C2} = 78.13 \), \( c_{S3,C3} = 99.02 \), \( c_{S3,C4} = 277.07 \), \( c_{S3,C5} = 884.45 \), \( c_{S3,C6} = 1800.86 \)
  - \( c_{S4,C1} = 868.75 \), \( c_{S4,C2} = 94.2 \), \( c_{S4,C3} = 1776.34 \), \( c_{S4,C4} = 285.48 \), \( c_{S4,C5} = 868.85 \), \( c_{S4,C6} = 86.55 \)
  - \( c_{S5,C1} = 1577 \), \( c_{S5,C2} = 760.15 \), \( c_{S5,C3} = 2090.19 \), \( c_{S5,C4} = 43.2 \), \( c_{S5,C5} = 1577.12 \), \( c_{S5,C6} = 1095.17 \)
  - \( c_{S6,C1} = 49.14 \), \( c_{S6,C2} = 4.33 \), \( c_{S6,C3} = 2079.57 \), \( c_{S6,C4} = 277.04 \), \( c_{S6,C5} = 1032.01 \), \( c_{S6,C6} = 1543.49 \)

Decision Variables:
- \( y_i \in \{0,1\} \) for \( i \in I \): 1 if supplier \( i \) is activated, 0 otherwise.
- \( x_{ij} \geq 0 \) for \( i \in I, j \in J \): quantity supplied from supplier \( i \) to store \( j \).

Mathematical Model:

\[
\begin{align*}
\textbf{Objective:} \quad & \min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\textbf{Subject to:} \\
& \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& x_{ij} \leq d_j y_i \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I, j \in J \\
\end{align*}
\]

Where:
- \( f_i \) is the fixed cost for supplier \( i \).
- \( c_{ij} \) is the transportation cost per unit from supplier \( i \) to store \( j \).
- \( d_j \) is the demand at store \( j \).

All parameters are explicitly listed above. This model determines which suppliers to activate and how much each should supply to each store to minimize the total cost while meeting all store demands.