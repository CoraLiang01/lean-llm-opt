Let:
- \( I = \{1,2,\ldots,10\} \) be the set of potential warehouse locations, corresponding to warehouses W1 through W10.
- \( J = \{1,2,\ldots,20\} \) be the set of customers, corresponding to customers C1 through C20.

Parameters:
- Fixed annual opening costs for each warehouse:
  \[
  f = [2000, 2500, 1800, 3200, 1500, 4000, 2800, 1950, 3500, 2200]
  \]
  where \( f_i \) is the fixed cost for warehouse \( i \) (W1 is 1, W2 is 2, etc.).

- Maximum service capacities for each warehouse:
  \[
  K = [1000, 1500, 1200, 2000, 800, 2500, 1800, 1100, 2100, 1300]
  \]
  where \( K_i \) is the capacity for warehouse \( i \).

- Customer demands:
  \[
  d = [800, 600, 500, 700, 450, 950, 350, 850, 400, 750, 900, 550, 650, 820, 480, 920, 320, 780, 520, 680]
  \]
  where \( d_j \) is the demand for customer \( j \) (C1 is 1, C2 is 2, etc.).

- Transportation costs from warehouse \( i \) to customer \( j \) (matrix \( c_{ij} \)):
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
  where row \( i \) corresponds to warehouse \( i \), and column \( j \) to customer \( j \).

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if warehouse \( i \) is opened, 0 otherwise.
- \( x_{ij} \geq 0 \): amount of customer \( j \)'s demand served from warehouse \( i \).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^{10} f_i y_i + \sum_{i=1}^{10} \sum_{j=1}^{20} c_{ij} x_{ij} \\
\text{subject to} \quad
& \sum_{i=1}^{10} x_{ij} = d_j \quad \forall j = 1,\ldots,20 \quad \text{(all customer demand must be met)} \\
& \sum_{j=1}^{20} x_{ij} \leq K_i y_i \quad \forall i = 1,\ldots,10 \quad \text{(do not exceed warehouse capacity; only if open)} \\
& x_{ij} \geq 0 \quad \forall i = 1,\ldots,10, \; j = 1,\ldots,20 \\
& y_i \in \{0,1\} \quad \forall i = 1,\ldots,10
\end{align*}
\]

Where:
- \( f_i \) is the fixed cost for warehouse \( i \) (see vector above),
- \( K_i \) is the capacity for warehouse \( i \) (see vector above),
- \( d_j \) is the demand for customer \( j \) (see vector above),
- \( c_{ij} \) is the transportation cost per unit from warehouse \( i \) to customer \( j \) (see matrix above).

This model selects the optimal subset of warehouses to open and assigns customer demand to minimize the total cost, subject to capacity and demand fulfillment constraints.