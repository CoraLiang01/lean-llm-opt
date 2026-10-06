Let:
- \( I = \{1,2,\ldots,10\} \) index set of warehouses, corresponding to W1–W10.
- \( J = \{1,2,\ldots,20\} \) index set of customers, corresponding to C1–C20.

Parameters:
- Warehouse IDs: W1, W2, W3, W4, W5, W6, W7, W8, W9, W10
- Customer IDs: C1, C2, ..., C20

- Fixed annual opening costs for each warehouse (\( f_i \)):
  - \( f_1 = 2000 \)
  - \( f_2 = 2500 \)
  - \( f_3 = 1800 \)
  - \( f_4 = 3200 \)
  - \( f_5 = 1500 \)
  - \( f_6 = 4000 \)
  - \( f_7 = 2800 \)
  - \( f_8 = 1950 \)
  - \( f_9 = 3500 \)
  - \( f_{10} = 2200 \)

- Maximum service capacities for each warehouse (\( s_i \)):
  - \( s_1 = 1000 \)
  - \( s_2 = 1500 \)
  - \( s_3 = 1200 \)
  - \( s_4 = 2000 \)
  - \( s_5 = 800 \)
  - \( s_6 = 2500 \)
  - \( s_7 = 1800 \)
  - \( s_8 = 1100 \)
  - \( s_9 = 2100 \)
  - \( s_{10} = 1300 \)

- Customer demands (\( d_j \)):
  - \( d_1 = 800 \)
  - \( d_2 = 600 \)
  - \( d_3 = 500 \)
  - \( d_4 = 700 \)
  - \( d_5 = 450 \)
  - \( d_6 = 950 \)
  - \( d_7 = 350 \)
  - \( d_8 = 850 \)
  - \( d_9 = 400 \)
  - \( d_{10} = 750 \)
  - \( d_{11} = 900 \)
  - \( d_{12} = 550 \)
  - \( d_{13} = 650 \)
  - \( d_{14} = 820 \)
  - \( d_{15} = 480 \)
  - \( d_{16} = 920 \)
  - \( d_{17} = 320 \)
  - \( d_{18} = 780 \)
  - \( d_{19} = 520 \)
  - \( d_{20} = 680 \)

- Transportation costs (\( c_{ij} \)), where \( i \) is warehouse index (1–10), \( j \) is customer index (1–20):

\[
C = \begin{bmatrix}
10 & 15 & 20 & 11 & 16 & 18 & 7 & 12 & 22 & 9 & 14 & 19 & 25 & 13 & 17 & 6 & 21 & 15 & 8 & 10 \\
18 & 12 & 9 & 14 & 10 & 5 & 19 & 23 & 11 & 16 & 20 & 8 & 15 & 22 & 7 & 13 & 24 & 17 & 12 & 6 \\
13 & 17 & 15 & 8 & 12 & 21 & 16 & 10 & 5 & 24 & 13 & 22 & 7 & 19 & 14 & 18 & 9 & 25 & 11 & 16 \\
7 & 22 & 11 & 16 & 20 & 8 & 15 & 19 & 13 & 25 & 6 & 14 & 21 & 9 & 23 & 17 & 10 & 18 & 24 & 5 \\
16 & 9 & 25 & 13 & 7 & 10 & 23 & 14 & 18 & 21 & 5 & 17 & 9 & 24 & 12 & 20 & 6 & 15 & 19 & 11 \\
22 & 6 & 14 & 19 & 23 & 11 & 8 & 17 & 9 & 12 & 15 & 24 & 5 & 20 & 10 & 25 & 13 & 7 & 18 & 16 \\
8 & 25 & 17 & 9 & 14 & 22 & 11 & 6 & 16 & 20 & 18 & 13 & 24 & 5 & 19 & 12 & 23 & 10 & 7 & 15 \\
19 & 11 & 7 & 21 & 15 & 24 & 13 & 16 & 20 & 8 & 17 & 10 & 12 & 23 & 5 & 14 & 22 & 9 & 16 & 25 \\
12 & 20 & 5 & 23 & 17 & 14 & 9 & 25 & 18 & 11 & 16 & 21 & 10 & 7 & 24 & 15 & 19 & 6 & 13 & 22 \\
25 & 14 & 22 & 5 & 19 & 12 & 24 & 7 & 15 & 17 & 23 & 6 & 16 & 10 & 20 & 9 & 18 & 11 & 25 & 14 \\
\end{bmatrix}
\]

Variables:
- \( y_i \in \{0,1\} \): 1 if warehouse \( i \) is opened, 0 otherwise.
- \( x_{ij} \geq 0 \): amount of customer \( j \)'s demand served from warehouse \( i \).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^{10} f_i y_i + \sum_{i=1}^{10} \sum_{j=1}^{20} c_{ij} x_{ij} \\
\text{subject to} \quad
& \sum_{i=1}^{10} x_{ij} = d_j \quad \forall j = 1,\ldots,20 \quad \text{(all customer demand must be met)} \\
& \sum_{j=1}^{20} x_{ij} \leq s_i y_i \quad \forall i = 1,\ldots,10 \quad \text{(do not exceed warehouse capacity; only if open)} \\
& x_{ij} \geq 0 \quad \forall i, j \\
& y_i \in \{0,1\} \quad \forall i
\end{align*}
\]

Where:
- \( f_i \): fixed cost for warehouse \( i \) (see above)
- \( s_i \): capacity for warehouse \( i \) (see above)
- \( d_j \): demand for customer \( j \) (see above)
- \( c_{ij} \): transportation cost from warehouse \( i \) to customer \( j \) (see matrix above)

This is a capacitated facility location problem with full parameter specification as per the provided data.