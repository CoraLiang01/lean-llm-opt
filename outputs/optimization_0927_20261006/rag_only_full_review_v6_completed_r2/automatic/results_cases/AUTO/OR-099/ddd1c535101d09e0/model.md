Let:
- \( I = \{1,2,3,4,5,6,7,8,9,10,11\} \) be the set of potential warehouses.
- \( J = \{1,2,3,4,5,6,7,8,9,10,11\} \) be the set of stores.

Parameters:
- Opening cost for warehouse \( i \): \( f_i \)
  - \( f = [3000, 3200, 3100, 2800, 3500, 2700, 2900, 3050, 3100, 2200, 2890] \)
- Capacity of warehouse \( i \): \( cap_i \)
  - \( cap = [180, 160, 200, 150, 170, 190, 160, 175, 170, 180, 190] \)
- Demand of store \( j \): \( d_j \)
  - \( d = [30, 40, 20, 35, 20, 25, 45, 38, 32, 41, 44] \)
- Transportation cost from warehouse \( i \) to store \( j \): \( c_{ij} \), given by the following matrix (rows: warehouses 1–11, columns: stores 1–11):

\[
C = \begin{bmatrix}
12 & 11 & 14 & 15 & 17 & 13 & 12 & 16 & 16 & 14 & 15 \\
17 & 19 & 15 & 20 & 18 & 14 & 17 & 15 & 13 & 15 & 16 \\
13 & 14 & 12 & 14 & 16 & 15 & 11 & 14 & 16 & 18 & 17 \\
18 & 16 & 17 & 13 & 18 & 17 & 14 & 19 & 16 & 13 & 18 \\
10 & 13 & 12 & 19 & 15 & 11 & 12 & 14 & 12 & 15 & 17 \\
15 & 12 & 14 & 16 & 13 & 17 & 16 & 16 & 14 & 18 & 19 \\
14 & 13 & 15 & 17 & 12 & 13 & 14 & 15 & 12 & 16 & 14 \\
19 & 16 & 18 & 20 & 17 & 19 & 16 & 18 & 15 & 15 & 18 \\
17 & 18 & 12 & 14 & 16 & 15 & 14 & 17 & 21 & 15 & 18 \\
14 & 13 & 15 & 17 & 16 & 18 & 14 & 19 & 15 & 17 & 19 \\
15 & 13 & 16 & 17 & 11 & 13 & 14 & 15 & 19 & 21 & 13 \\
\end{bmatrix}
\]

Decision variables:
- \( y_i \in \{0,1\} \): 1 if warehouse \( i \) is opened, 0 otherwise.
- \( x_{ij} \geq 0 \): amount supplied from warehouse \( i \) to store \( j \).

Objective:
\[
\min \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction for each store:
\[
\sum_{i=1}^{11} x_{ij} = d_j \quad \forall j \in J
\]

2. Warehouse capacity:
\[
\sum_{j=1}^{11} x_{ij} \leq cap_i \cdot y_i \quad \forall i \in I
\]

3. Non-negativity and binary constraints:
\[
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]
\[
y_i \in \{0,1\} \quad \forall i \in I
\]

Where:
- \( f = [3000, 3200, 3100, 2800, 3500, 2700, 2900, 3050, 3100, 2200, 2890] \)
- \( cap = [180, 160, 200, 150, 170, 190, 160, 175, 170, 180, 190] \)
- \( d = [30, 40, 20, 35, 20, 25, 45, 38, 32, 41, 44] \)
- \( C \) as above.

This is a standard capacitated facility location problem (CFLP) with the given data.