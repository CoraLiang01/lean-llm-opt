Let:
- I = {S1, S2, S3, S4, S5, S6} be the set of suppliers.
- J = {C1, C2, C3, C4, C5, C6} be the set of stores.

Parameters:
- Fixed costs for each supplier:
  - f = [f_S1, f_S2, f_S3, f_S4, f_S5, f_S6] = [98.88, 99.73, 94.01, 93.77, 107.59, 112.65]
- Transportation costs per unit from supplier i to store j (matrix c_{ij}):

|      | C1     | C2     | C3      | C4      | C5     | C6      |
|------|--------|--------|---------|---------|--------|---------|
| S1   | 0.08   | 52.33  | 73.57   | 1237.33 | 0.07   | 112.16  |
| S2   | 46.02  | 175.23 | 2026.83 | 299.89  | 966.53 | 1590.42 |
| S3   | 1031.74| 78.13  | 99.02   | 277.07  | 884.45 | 1800.86 |
| S4   | 868.75 | 94.20  | 1776.34 | 285.48  | 868.85 | 86.55   |
| S5   | 1577   | 760.15 | 2090.19 | 43.20   | 1577.12| 1095.17 |
| S6   | 49.14  | 4.33   | 2079.57 | 277.04  | 1032.01| 1543.49 |

- Demand at each store:
  - d = [d_C1, d_C2, d_C3, d_C4, d_C5, d_C6] = [216, 216, 216, 144, 144, 144]

Decision variables:
- y_i ∈ {0,1} for each supplier i ∈ I, where y_i = 1 if supplier i is activated, 0 otherwise.
- x_{ij} ≥ 0 for each supplier i ∈ I and store j ∈ J, representing the quantity shipped from supplier i to store j.

Model:

Minimize total cost:
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where:
- \( f_i \) is the fixed cost for supplier i,
- \( c_{ij} \) is the transportation cost per unit from supplier i to store j,
- \( x_{ij} \) is the quantity shipped from supplier i to store j,
- \( y_i \) is the binary activation variable for supplier i.

Subject to:

1. Demand satisfaction at each store:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]
where \( d_j \) is the demand at store j.

2. Logical link between supplier activation and shipments:
\[
\sum_{j \in J} x_{ij} \leq M \cdot y_i \quad \forall i \in I
\]
where M is a sufficiently large constant (e.g., \( M = \sum_{j \in J} d_j = 1080 \)), ensuring that if y_i = 0, then x_{ij} = 0 for all j.

3. Non-negativity and integrality:
\[
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]
\[
y_i \in \{0,1\} \quad \forall i \in I
\]

Parameter values:

- I = {S1, S2, S3, S4, S5, S6}
- J = {C1, C2, C3, C4, C5, C6}
- f = [98.88, 99.73, 94.01, 93.77, 107.59, 112.65]
- c =

\[
\begin{bmatrix}
0.08 & 52.33 & 73.57 & 1237.33 & 0.07 & 112.16 \\
46.02 & 175.23 & 2026.83 & 299.89 & 966.53 & 1590.42 \\
1031.74 & 78.13 & 99.02 & 277.07 & 884.45 & 1800.86 \\
868.75 & 94.20 & 1776.34 & 285.48 & 868.85 & 86.55 \\
1577 & 760.15 & 2090.19 & 43.20 & 1577.12 & 1095.17 \\
49.14 & 4.33 & 2079.57 & 277.04 & 1032.01 & 1543.49 \\
\end{bmatrix}
\]

- d = [216, 216, 216, 144, 144, 144]
- M = 1080

Variables:
- y_i ∈ {0,1} for i = 1,...,6
- x_{ij} ≥ 0 for i = 1,...,6; j = 1,...,6

This model determines which suppliers to activate and how much each should ship to each store to minimize the total cost while meeting all store demands.