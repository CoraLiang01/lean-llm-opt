Let:
- \( I = \{1, 2, ..., 11\} \) be the set of potential warehouses.
- \( J = \{1, 2, ..., 11\} \) be the set of stores.

Parameters:
- \( f_i \): Opening cost for warehouse \( i \).
- \( s_i \): Capacity of warehouse \( i \).
- \( d_j \): Demand of store \( j \).
- \( c_{ij} \): Transportation cost per unit from warehouse \( i \) to store \( j \).

From the data:

Warehouses (i), Opening Costs (\( f_i \)), and Capacities (\( s_i \)):
\[
\begin{align*}
&\text{Warehouse 1:} \quad f_1 = 3000, \quad s_1 = 180 \\
&\text{Warehouse 2:} \quad f_2 = 3200, \quad s_2 = 160 \\
&\text{Warehouse 3:} \quad f_3 = 3100, \quad s_3 = 200 \\
&\text{Warehouse 4:} \quad f_4 = 2800, \quad s_4 = 150 \\
&\text{Warehouse 5:} \quad f_5 = 3500, \quad s_5 = 170 \\
&\text{Warehouse 6:} \quad f_6 = 2700, \quad s_6 = 190 \\
&\text{Warehouse 7:} \quad f_7 = 2900, \quad s_7 = 160 \\
&\text{Warehouse 8:} \quad f_8 = 3050, \quad s_8 = 175 \\
&\text{Warehouse 9:} \quad f_9 = 3100, \quad s_9 = 170 \\
&\text{Warehouse 10:} \quad f_{10} = 2200, \quad s_{10} = 180 \\
&\text{Warehouse 11:} \quad f_{11} = 2890, \quad s_{11} = 190 \\
\end{align*}
\]

Stores (j) and Demands (\( d_j \)):
\[
\begin{align*}
&\text{Store 1:} \quad d_1 = 30 \\
&\text{Store 2:} \quad d_2 = 40 \\
&\text{Store 3:} \quad d_3 = 20 \\
&\text{Store 4:} \quad d_4 = 35 \\
&\text{Store 5:} \quad d_5 = 20 \\
&\text{Store 6:} \quad d_6 = 25 \\
&\text{Store 7:} \quad d_7 = 45 \\
&\text{Store 8:} \quad d_8 = 38 \\
&\text{Store 9:} \quad d_9 = 32 \\
&\text{Store 10:} \quad d_{10} = 41 \\
&\text{Store 11:} \quad d_{11} = 44 \\
\end{align*}
\]

Transportation Cost Matrix (\( c_{ij} \)), where \( i \) is the warehouse (row), \( j \) is the store (column):

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

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if warehouse \( i \) is opened, 0 otherwise.
- \( x_{ij} \geq 0 \): Amount supplied from warehouse \( i \) to store \( j \).

Mathematical Model:

\[
\begin{align*}
\textbf{Minimize:} \quad & \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij} \\
\\
\textbf{Subject to:} \\
& \sum_{i=1}^{11} x_{ij} = d_j \quad \forall j = 1, ..., 11 \quad \text{(meet each store's demand)} \\
& \sum_{j=1}^{11} x_{ij} \leq s_i y_i \quad \forall i = 1, ..., 11 \quad \text{(do not exceed warehouse capacity)} \\
& x_{ij} \geq 0 \quad \forall i, j \\
& y_i \in \{0,1\} \quad \forall i \\
\end{align*}
\]

Where:
- \( f_i \), \( s_i \), \( d_j \), and \( c_{ij} \) are as specified above.

This model determines which warehouses to open and how to assign store demands to minimize the total cost, while satisfying all constraints.