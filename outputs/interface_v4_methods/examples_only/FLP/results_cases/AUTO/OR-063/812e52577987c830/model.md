Let:
- \( F = \{S1, S2, S3, S4, S5, S6, S7\} \) be the set of warehouses (facilities).
- \( C = \{C1, C2, C3, C4, C5, C6, C7\} \) be the set of musicians/bands (customers).

Parameters:
- Fixed cost for opening warehouse \( i \):  
  \[
  f = 
  \begin{bmatrix}
  102.33 \\
  94.92 \\
  91.83 \\
  98.71 \\
  95.73 \\
  99.96 \\
  98.16 \\
  \end{bmatrix}
  \]
  where \( f_i \) corresponds to warehouse \( S_i \) for \( i = 1, \ldots, 7 \).

- Demand for each musician/band \( j \):  
  \[
  d = 
  \begin{bmatrix}
  1083 \\
  776 \\
  16214 \\
  553 \\
  17106 \\
  594 \\
  732 \\
  \end{bmatrix}
  \]
  where \( d_j \) corresponds to customer \( C_j \) for \( j = 1, \ldots, 7 \).

- Transportation cost per unit from warehouse \( i \) to customer \( j \):  
  \[
  c = 
  \begin{bmatrix}
  1506.22 & 70.90   & 8.44    & 260.27  & 197.47  & 71.71   & 61.19   \\
  1732.65 & 1780.72 & 567.44  & 448.68  & 29.00   & 1484.91 & 963.92  \\
  115.66  & 100.76  & 64.68   & 1324.53 & 64.99   & 134.88  & 2102.83 \\
  1254.78 & 1115.63 & 52.31   & 1036.16 & 892.63  & 1464.04 & 1383.41 \\
  42.90   & 891.01  & 1013.94 & 1128.72 & 58.91   & 42.89   & 1570.31 \\
  0.70    & 139.46  & 70.03   & 79.15   & 1482.00 & 0.91    & 110.46  \\
  1732.30 & 1780.44 & 486.50  & 523.74  & 522.08  & 82.48   & 826.41  \\
  \end{bmatrix}
  \]
  where row \( i \) corresponds to warehouse \( S_i \), column \( j \) to customer \( C_j \).

Decision variables:
- \( y_i \in \{0,1\} \): 1 if warehouse \( S_i \) is opened, 0 otherwise.
- \( x_{ij} \geq 0 \): quantity of goods supplied from warehouse \( S_i \) to customer \( C_j \).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^{7} f_i y_i + \sum_{i=1}^{7} \sum_{j=1}^{7} c_{ij} x_{ij} \\
\text{subject to} \quad
& \sum_{i=1}^{7} x_{ij} = d_j, \quad \forall j = 1, \ldots, 7 \\
& x_{ij} \leq d_j y_i, \quad \forall i = 1, \ldots, 7; \; j = 1, \ldots, 7 \\
& y_i \in \{0,1\}, \quad \forall i = 1, \ldots, 7 \\
& x_{ij} \geq 0, \quad \forall i = 1, \ldots, 7; \; j = 1, \ldots, 7 \\
\end{align*}
\]

Where:
- \( f_i \) is the fixed cost for warehouse \( S_i \) (see vector above).
- \( c_{ij} \) is the transportation cost per unit from warehouse \( S_i \) to customer \( C_j \) (see matrix above).
- \( d_j \) is the demand for customer \( C_j \) (see vector above).

All parameters, vectors, and matrices are explicitly stated as retrieved from the CSV files. The objective is to minimize the total cost (fixed + transportation) while meeting all musician/band demands and only allowing supply from open warehouses.