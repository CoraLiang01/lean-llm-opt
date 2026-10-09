Let:
- \( I = \{1,2,\ldots,10\} \) be the set of potential warehouse locations, indexed by \( i \), corresponding to warehouses W1–W10.
- \( J = \{1,2,\ldots,20\} \) be the set of customers, indexed by \( j \), corresponding to customers C1–C20.

Parameters:
- Fixed annual opening cost for warehouse \( i \):  
  \( f_i \), where  
  \( f = [2000, 2500, 1800, 3200, 1500, 4000, 2800, 1950, 3500, 2200] \)  
  (W1–W10 order).

- Maximum service capacity for warehouse \( i \):  
  \( s_i \), where  
  \( s = [1000, 1500, 1200, 2000, 800, 2500, 1800, 1100, 2100, 1300] \)  
  (W1–W10 order).

- Demand of customer \( j \):  
  \( d_j \), where  
  \( d = [800, 600, 500, 700, 450, 950, 350, 850, 400, 750, 900, 550, 650, 820, 480, 920, 320, 780, 520, 680] \)  
  (C1–C20 order).

- Variable transportation cost from warehouse \( i \) to customer \( j \):  
  \( c_{ij} \), given by the following matrix (rows: W1–W10, columns: C1–C20):

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

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if warehouse \( i \) is opened, 0 otherwise.
- \( x_{ij} \geq 0 \): amount of customer \( j \)'s demand supplied from warehouse \( i \).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^{10} f_i y_i + \sum_{i=1}^{10} \sum_{j=1}^{20} c_{ij} x_{ij} \\
\text{subject to} \quad
& \sum_{i=1}^{10} x_{ij} = d_j \quad \forall j = 1,\ldots,20 \\
& \sum_{j=1}^{20} x_{ij} \leq s_i y_i \quad \forall i = 1,\ldots,10 \\
& x_{ij} \geq 0 \quad \forall i = 1,\ldots,10; \; j = 1,\ldots,20 \\
& y_i \in \{0,1\} \quad \forall i = 1,\ldots,10 \\
\end{align*}
\]

Where:
- \( f_i \), \( s_i \), \( d_j \), and \( c_{ij} \) are as specified above.
- The first constraint ensures all customer demand is satisfied.
- The second constraint ensures no warehouse exceeds its capacity and can only serve if opened.
- The objective is to minimize the sum of fixed opening costs and total transportation costs.

All parameters (vectors and matrices) are explicitly provided above.