Let:
- \( I = \{1, 2, ..., 11\} \) be the set of potential warehouses.
- \( J = \{1, 2, ..., 11\} \) be the set of stores.

Parameters:
- Opening cost for warehouse \( i \): \( f_i \)
- Capacity of warehouse \( i \): \( \text{cap}_i \)
- Demand of store \( j \): \( d_j \)
- Transportation cost from warehouse \( i \) to store \( j \): \( c_{ij} \)

From the data:

Warehouses, opening costs, and capacities:
\[
\begin{align*}
&\text{Warehouse 1: } f_1 = 3000, \quad \text{cap}_1 = 180 \\
&\text{Warehouse 2: } f_2 = 3200, \quad \text{cap}_2 = 160 \\
&\text{Warehouse 3: } f_3 = 3100, \quad \text{cap}_3 = 200 \\
&\text{Warehouse 4: } f_4 = 2800, \quad \text{cap}_4 = 150 \\
&\text{Warehouse 5: } f_5 = 3500, \quad \text{cap}_5 = 170 \\
&\text{Warehouse 6: } f_6 = 2700, \quad \text{cap}_6 = 190 \\
&\text{Warehouse 7: } f_7 = 2900, \quad \text{cap}_7 = 160 \\
&\text{Warehouse 8: } f_8 = 3050, \quad \text{cap}_8 = 175 \\
&\text{Warehouse 9: } f_9 = 3100, \quad \text{cap}_9 = 170 \\
&\text{Warehouse 10: } f_{10} = 2200, \quad \text{cap}_{10} = 180 \\
&\text{Warehouse 11: } f_{11} = 2890, \quad \text{cap}_{11} = 190 \\
\end{align*}
\]

Stores and demands:
\[
\begin{align*}
&\text{Store 1: } d_1 = 30 \\
&\text{Store 2: } d_2 = 40 \\
&\text{Store 3: } d_3 = 20 \\
&\text{Store 4: } d_4 = 35 \\
&\text{Store 5: } d_5 = 20 \\
&\text{Store 6: } d_6 = 25 \\
&\text{Store 7: } d_7 = 45 \\
&\text{Store 8: } d_8 = 38 \\
&\text{Store 9: } d_9 = 32 \\
&\text{Store 10: } d_{10} = 41 \\
&\text{Store 11: } d_{11} = 44 \\
\end{align*}
\]

Transportation cost matrix \( c_{ij} \) (rows: warehouses 1–11, columns: stores 1–11):

\[
\begin{array}{c|ccccccccccc}
 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 \\
\hline
1 & 12 & 11 & 14 & 15 & 17 & 13 & 12 & 16 & 16 & 14 & 15 \\
2 & 17 & 19 & 15 & 20 & 18 & 14 & 17 & 15 & 13 & 15 & 16 \\
3 & 13 & 14 & 12 & 14 & 16 & 15 & 11 & 14 & 16 & 18 & 17 \\
4 & 18 & 16 & 17 & 13 & 18 & 17 & 14 & 19 & 16 & 13 & 18 \\
5 & 10 & 13 & 12 & 19 & 15 & 11 & 12 & 14 & 12 & 15 & 17 \\
6 & 15 & 12 & 14 & 16 & 13 & 17 & 16 & 16 & 14 & 18 & 19 \\
7 & 14 & 13 & 15 & 17 & 12 & 13 & 14 & 15 & 12 & 16 & 14 \\
8 & 19 & 16 & 18 & 20 & 17 & 19 & 16 & 18 & 15 & 15 & 18 \\
9 & 17 & 18 & 12 & 14 & 16 & 15 & 14 & 17 & 21 & 15 & 18 \\
10 & 14 & 13 & 15 & 17 & 16 & 18 & 14 & 19 & 15 & 17 & 19 \\
11 & 15 & 13 & 16 & 17 & 11 & 13 & 14 & 15 & 19 & 21 & 13 \\
\end{array}
\]

Decision variables:
- \( y_i \in \{0,1\} \): 1 if warehouse \( i \) is opened, 0 otherwise.
- \( x_{ij} \geq 0 \): amount supplied from warehouse \( i \) to store \( j \).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij} \\
\text{subject to} \quad
& \sum_{i=1}^{11} x_{ij} = d_j \quad \forall j = 1, \ldots, 11 \\
& \sum_{j=1}^{11} x_{ij} \leq \text{cap}_i y_i \quad \forall i = 1, \ldots, 11 \\
& x_{ij} \geq 0 \quad \forall i, j \\
& y_i \in \{0,1\} \quad \forall i \\
\end{align*}
\]

Where:
- \( f_i \) and \( \text{cap}_i \) are as listed above for each warehouse \( i \).
- \( d_j \) is as listed above for each store \( j \).
- \( c_{ij} \) is as in the matrix above for each warehouse \( i \) and store \( j \).

This model determines which warehouses to open and how to assign store demands to minimize the total cost, while satisfying all constraints.