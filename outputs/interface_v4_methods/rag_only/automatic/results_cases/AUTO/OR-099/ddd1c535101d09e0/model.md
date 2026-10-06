Let:
- \( I = \{1, 2, ..., 11\} \) be the set of potential warehouses.
- \( J = \{1, 2, ..., 11\} \) be the set of stores.

Parameters:
- \( f_i \): Opening cost for warehouse \( i \).
- \( s_i \): Capacity of warehouse \( i \).
- \( d_j \): Demand of store \( j \).
- \( c_{ij} \): Transportation cost per unit from warehouse \( i \) to store \( j \).

From the data:

Warehouses (i = 1..11):

| i  | \( f_i \) | \( s_i \) |
|----|-----------|-----------|
| 1  | 3000      | 180       |
| 2  | 3200      | 160       |
| 3  | 3100      | 200       |
| 4  | 2800      | 150       |
| 5  | 3500      | 170       |
| 6  | 2700      | 190       |
| 7  | 2900      | 160       |
| 8  | 3050      | 175       |
| 9  | 3100      | 170       |
| 10 | 2200      | 180       |
| 11 | 2890      | 190       |

Stores (j = 1..11):

| j  | \( d_j \) |
|----|-----------|
| 1  | 30        |
| 2  | 40        |
| 3  | 20        |
| 4  | 35        |
| 5  | 20        |
| 6  | 25        |
| 7  | 45        |
| 8  | 38        |
| 9  | 32        |
| 10 | 41        |
| 11 | 44        |

Transportation cost matrix \( c_{ij} \) (i = 1..11, j = 1..11):

\[
c_{ij} =
\begin{bmatrix}
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
- \( x_{ij} \geq 0 \): Amount supplied from warehouse \( i \) to store \( j \).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij} \\
\text{subject to} \quad
& \sum_{i=1}^{11} x_{ij} = d_j \quad \forall j = 1, \ldots, 11 \\
& \sum_{j=1}^{11} x_{ij} \leq s_i y_i \quad \forall i = 1, \ldots, 11 \\
& x_{ij} \geq 0 \quad \forall i, j \\
& y_i \in \{0,1\} \quad \forall i
\end{align*}
\]

Where:
- \( f_i \), \( s_i \), \( d_j \), and \( c_{ij} \) are as specified above.

This model determines which warehouses to open and how to assign store demands to minimize the total cost, while satisfying all constraints.