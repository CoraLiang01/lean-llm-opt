Let:
- \( I = \{S1, S2, S3, S4, S5, S6\} \) be the set of suppliers.
- \( J = \{C1, C2, C3, C4, C5, C6\} \) be the set of stores.

Parameters:
- Fixed costs for each supplier \( i \in I \):
  - \( f_{S1} = 98.88 \)
  - \( f_{S2} = 99.73 \)
  - \( f_{S3} = 94.01 \)
  - \( f_{S4} = 93.77 \)
  - \( f_{S5} = 107.59 \)
  - \( f_{S6} = 112.65 \)

- Demand for each store \( j \in J \):
  - \( d_{C1} = 216 \)
  - \( d_{C2} = 216 \)
  - \( d_{C3} = 216 \)
  - \( d_{C4} = 144 \)
  - \( d_{C5} = 144 \)
  - \( d_{C6} = 144 \)

- Transportation costs per unit from supplier \( i \) to store \( j \) (\( c_{ij} \)):

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
- \( y_i \in \{0,1\} \) for each \( i \in I \): 1 if supplier \( i \) is operational, 0 otherwise.
- \( x_{ij} \geq 0 \) for each \( i \in I, j \in J \): quantity supplied from supplier \( i \) to store \( j \).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{Subject to:} \quad & \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M_i y_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]

Where:
- \( f_i \) is the fixed cost for supplier \( i \) (see above).
- \( c_{ij} \) is the transportation cost per unit from supplier \( i \) to store \( j \) (see table above).
- \( d_j \) is the demand for store \( j \) (see above).
- \( M_i \) is a sufficiently large number (e.g., \( M_i = \sum_{j \in J} d_j \)), ensuring that if \( y_i = 0 \), then \( x_{ij} = 0 \) for all \( j \).

All parameters, vectors, and matrices are explicitly stated above. The objective is to minimize the total cost (fixed + transportation) while meeting all store demands and only allowing shipments from open suppliers.