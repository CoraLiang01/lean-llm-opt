Let:
- I = {W1, W2, W3, W4, W5, W6, W7, W8, W9, W10} be the set of potential warehouses.
- J = {C1, C2, ..., C20} be the set of customers.

Parameters:
- Fixed opening cost for warehouse i ∈ I: 
  f_i = [2000, 2500, 1800, 3200, 1500, 4000, 2800, 1950, 3500, 2200] for i = W1, ..., W10 (in order).
- Capacity of warehouse i ∈ I: 
  cap_i = [1000, 1500, 1200, 2000, 800, 2500, 1800, 1100, 2100, 1300] for i = W1, ..., W10 (in order).
- Demand of customer j ∈ J: 
  d_j = [800, 600, 500, 700, 450, 950, 350, 850, 400, 750, 900, 550, 650, 820, 480, 920, 320, 780, 520, 680] for j = C1, ..., C20 (in order).
- Transportation cost from warehouse i to customer j: 
  c_{ij} is given by the following 10×20 matrix (rows: W1–W10, columns: C1–C20):

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

Decision variables:
- y_i ∈ {0,1} for each warehouse i ∈ I, where y_i = 1 if warehouse i is opened, 0 otherwise.
- x_{ij} ≥ 0 for each warehouse i ∈ I and customer j ∈ J, representing the amount of customer j's demand served from warehouse i.

Mathematical Model:

Minimize total cost:
\[
\text{Minimize} \quad Z = \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand fulfillment for each customer:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]

2. Warehouse capacity:
\[
\sum_{j \in J} x_{ij} \leq cap_i y_i \quad \forall i \in I
\]

3. Non-negativity and integrality:
\[
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]
\[
y_i \in \{0,1\} \quad \forall i \in I
\]

Where:
- I = {W1, W2, ..., W10}
- J = {C1, C2, ..., C20}
- f_i, cap_i, d_j, and c_{ij} are as specified above.

This model selects the optimal subset of warehouses to open and assigns customer demand to minimize the sum of fixed opening and transportation costs, subject to capacity and demand constraints.