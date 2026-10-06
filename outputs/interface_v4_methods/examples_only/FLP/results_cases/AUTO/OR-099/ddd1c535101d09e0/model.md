Let:
- \( I = \{1, 2, ..., 11\} \) be the set of potential warehouses.
- \( J = \{1, 2, ..., 11\} \) be the set of stores.

Parameters (from the CSV files):

Warehouse opening costs and capacities (from PotentialWarehouses_Costs.csv):

\[
\begin{array}{c|c|c}
\text{Warehouse } (i) & \text{Opening Cost } (f_i) & \text{Capacity } (K_i) \\
\hline
1 & 3000 & 180 \\
2 & 3200 & 160 \\
3 & 3100 & 200 \\
4 & 2800 & 150 \\
5 & 3500 & 170 \\
6 & 2700 & 190 \\
7 & 2900 & 160 \\
8 & 3050 & 175 \\
9 & 3100 & 170 \\
10 & 2200 & 180 \\
11 & 2890 & 190 \\
\end{array}
\]

Store demands (from Stores_Demands.csv):

\[
\begin{array}{c|c}
\text{Store } (j) & \text{Demand } (d_j) \\
\hline
1 & 30 \\
2 & 40 \\
3 & 20 \\
4 & 35 \\
5 & 20 \\
6 & 25 \\
7 & 45 \\
8 & 38 \\
9 & 32 \\
10 & 41 \\
11 & 44 \\
\end{array}
\]

Transportation costs \( c_{ij} \) (from TransportationCost.csv):

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
where \( C_{ij} \) is the cost from warehouse \( i \) to store \( j \), with \( i, j = 1, ..., 11 \).

Decision variables:
- \( y_i \in \{0,1\} \): 1 if warehouse \( i \) is opened, 0 otherwise.
- \( x_{ij} \geq 0 \): amount supplied from warehouse \( i \) to store \( j \).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij} \\
\text{subject to} \quad
& \sum_{i=1}^{11} x_{ij} = d_j \quad \forall j = 1, ..., 11 \\
& \sum_{j=1}^{11} x_{ij} \leq K_i y_i \quad \forall i = 1, ..., 11 \\
& x_{ij} \geq 0 \quad \forall i, j \\
& y_i \in \{0,1\} \quad \forall i \\
\end{align*}
\]

Where:
- \( f_i \) and \( K_i \) are as listed above for each warehouse.
- \( d_j \) is as listed above for each store.
- \( c_{ij} \) is as given in the transportation cost matrix above.

This model determines which warehouses to open and how to assign store demands to minimize the total cost, while satisfying all constraints.