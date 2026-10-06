Let:
- \( I = \{1,2,3,4,5,6\} \) be the set of suppliers, corresponding to S1–S6.
- \( J = \{1,2,3,4,5,6\} \) be the set of stores, corresponding to C1–C6.

Parameters:
- Fixed costs for each supplier:
  - \( f_1 = 98.88 \) (S1)
  - \( f_2 = 99.73 \) (S2)
  - \( f_3 = 94.01 \) (S3)
  - \( f_4 = 93.77 \) (S4)
  - \( f_5 = 107.59 \) (S5)
  - \( f_6 = 112.65 \) (S6)

- Demand for each store:
  - \( d_1 = 216 \) (C1)
  - \( d_2 = 216 \) (C2)
  - \( d_3 = 216 \) (C3)
  - \( d_4 = 144 \) (C4)
  - \( d_5 = 144 \) (C5)
  - \( d_6 = 144 \) (C6)

- Transportation costs per unit from supplier \( i \) to store \( j \) (\( c_{ij} \)):
  - \( c_{11} = 0.08 \), \( c_{12} = 52.33 \), \( c_{13} = 73.57 \), \( c_{14} = 1237.33 \), \( c_{15} = 0.07 \), \( c_{16} = 112.16 \)
  - \( c_{21} = 46.02 \), \( c_{22} = 175.23 \), \( c_{23} = 2026.83 \), \( c_{24} = 299.89 \), \( c_{25} = 966.53 \), \( c_{26} = 1590.42 \)
  - \( c_{31} = 1031.74 \), \( c_{32} = 78.13 \), \( c_{33} = 99.02 \), \( c_{34} = 277.07 \), \( c_{35} = 884.45 \), \( c_{36} = 1800.86 \)
  - \( c_{41} = 868.75 \), \( c_{42} = 94.2 \), \( c_{43} = 1776.34 \), \( c_{44} = 285.48 \), \( c_{45} = 868.85 \), \( c_{46} = 86.55 \)
  - \( c_{51} = 1577 \), \( c_{52} = 760.15 \), \( c_{53} = 2090.19 \), \( c_{54} = 43.2 \), \( c_{55} = 1577.12 \), \( c_{56} = 1095.17 \)
  - \( c_{61} = 49.14 \), \( c_{62} = 4.33 \), \( c_{63} = 2079.57 \), \( c_{64} = 277.04 \), \( c_{65} = 1032.01 \), \( c_{66} = 1543.49 \)

Decision Variables:
- \( y_i \in \{0,1\} \) for \( i \in I \): 1 if supplier \( i \) is operational (open), 0 otherwise.
- \( x_{ij} \geq 0 \) for \( i \in I, j \in J \): quantity supplied from supplier \( i \) to store \( j \).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^6 f_i y_i + \sum_{i=1}^6 \sum_{j=1}^6 c_{ij} x_{ij} \\
= \quad & 98.88 y_1 + 99.73 y_2 + 94.01 y_3 + 93.77 y_4 + 107.59 y_5 + 112.65 y_6 \\
& + \Big[ 0.08 x_{11} + 52.33 x_{12} + 73.57 x_{13} + 1237.33 x_{14} + 0.07 x_{15} + 112.16 x_{16} \\
& + 46.02 x_{21} + 175.23 x_{22} + 2026.83 x_{23} + 299.89 x_{24} + 966.53 x_{25} + 1590.42 x_{26} \\
& + 1031.74 x_{31} + 78.13 x_{32} + 99.02 x_{33} + 277.07 x_{34} + 884.45 x_{35} + 1800.86 x_{36} \\
& + 868.75 x_{41} + 94.2 x_{42} + 1776.34 x_{43} + 285.48 x_{44} + 868.85 x_{45} + 86.55 x_{46} \\
& + 1577 x_{51} + 760.15 x_{52} + 2090.19 x_{53} + 43.2 x_{54} + 1577.12 x_{55} + 1095.17 x_{56} \\
& + 49.14 x_{61} + 4.33 x_{62} + 2079.57 x_{63} + 277.04 x_{64} + 1032.01 x_{65} + 1543.49 x_{66} \Big]
\end{align*}
\]

Subject to:

1. Demand satisfaction for each store:
   \[
   \sum_{i=1}^6 x_{ij} = d_j \quad \forall j \in \{1,2,3,4,5,6\}
   \]
   That is,
   \begin{align*}
   x_{1,1} + x_{2,1} + x_{3,1} + x_{4,1} + x_{5,1} + x_{6,1} &= 216 \\
   x_{1,2} + x_{2,2} + x_{3,2} + x_{4,2} + x_{5,2} + x_{6,2} &= 216 \\
   x_{1,3} + x_{2,3} + x_{3,3} + x_{4,3} + x_{5,3} + x_{6,3} &= 216 \\
   x_{1,4} + x_{2,4} + x_{3,4} + x_{4,4} + x_{5,4} + x_{6,4} &= 144 \\
   x_{1,5} + x_{2,5} + x_{3,5} + x_{4,5} + x_{5,5} + x_{6,5} &= 144 \\
   x_{1,6} + x_{2,6} + x_{3,6} + x_{4,6} + x_{5,6} + x_{6,6} &= 144 \\
   \end{align*}

2. Linking constraints (no supply from a closed supplier):
   \[
   x_{ij} \leq d_j y_i \quad \forall i \in \{1,2,3,4,5,6\}, \forall j \in \{1,2,3,4,5,6\}
   \]

3. Variable domains:
   \[
   y_i \in \{0,1\} \quad \forall i \in \{1,2,3,4,5,6\}
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in \{1,2,3,4,5,6\}, \forall j \in \{1,2,3,4,5,6\}
   \]

All parameters (fixed costs, demands, transportation costs) are explicitly listed above. This model determines which suppliers to activate and how much each should supply to each store to minimize the total cost while meeting all store demands.