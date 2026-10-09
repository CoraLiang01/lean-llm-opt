Let:
- \( I = \{1, 2, ..., 11\} \) be the set of potential warehouses.
- \( J = \{1, 2, ..., 11\} \) be the set of stores.

Parameters:
- \( f_i \): Opening cost for warehouse \( i \).
- \( s_i \): Capacity of warehouse \( i \).
- \( d_j \): Demand of store \( j \).
- \( c_{ij} \): Transportation cost per unit from warehouse \( i \) to store \( j \).

Decision variables:
- \( y_i \in \{0,1\} \): 1 if warehouse \( i \) is opened, 0 otherwise.
- \( x_{ij} \geq 0 \): Amount supplied from warehouse \( i \) to store \( j \).

Objective:
Minimize total cost (opening + transportation):
\[
\min \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij}
\]

Subject to:
1. Each store's demand is met:
\[
\sum_{i=1}^{11} x_{ij} = d_j \quad \forall j \in J
\]
2. Warehouse capacity is not exceeded:
\[
\sum_{j=1}^{11} x_{ij} \leq s_i y_i \quad \forall i \in I
\]
3. Non-negativity and binary constraints:
\[
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]
\[
y_i \in \{0,1\} \quad \forall i \in I
\]

Parameters (from CSVs):

Warehouses, opening costs, and capacities:
\[
\begin{array}{cccc}
i & f_i & s_i \\
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

Stores and demands:
\[
\begin{array}{cc}
j & d_j \\
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

Transportation cost matrix \( c_{ij} \) (rows: warehouses \( i \), columns: stores \( j \)):
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

Full mathematical model:

\[
\begin{align*}
\min \quad & 3000y_1 + 3200y_2 + 3100y_3 + 2800y_4 + 3500y_5 + 2700y_6 + 2900y_7 + 3050y_8 + 3100y_9 + 2200y_{10} + 2890y_{11} \\
& + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij} \\
\text{where } c_{ij} \text{ is as above, and:} \\
\text{subject to:} \\
& \sum_{i=1}^{11} x_{ij} = d_j \quad \forall j=1,\ldots,11 \\
& \sum_{j=1}^{11} x_{ij} \leq s_i y_i \quad \forall i=1,\ldots,11 \\
& x_{ij} \geq 0 \quad \forall i,j \\
& y_i \in \{0,1\} \quad \forall i
\end{align*}
\]

Where all parameters (\( f_i, s_i, d_j, c_{ij} \)) are as listed above. This model determines which warehouses to open and how to assign store demands to minimize total cost while satisfying all constraints.