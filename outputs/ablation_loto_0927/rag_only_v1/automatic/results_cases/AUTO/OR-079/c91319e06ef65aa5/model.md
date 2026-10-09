Let us define the mathematical model for ElectroTech Manufacturing's strategic network restructuring as a mixed-integer programming (MIP) problem.

Sets:
- Let I = {A1, A2, ..., A15} be the set of potential factory sites (indexed by i).
- Let J = {B1, B2, ..., B8} be the set of distribution centers (indexed by j).

Parameters:
- Fixed cost for opening factory i: \( f_i \)
- Capacity of factory i: \( cap_i \)
- Demand at distribution center j: \( d_j \)
- Variable shipping cost per unit from factory i to distribution center j: \( c_{ij} \)

From the CSV data:

Factory fixed costs and capacities:
\[
\begin{align*}
&f = [0, 175, 300, 375, 500, 200, 260, 220, 320, 280, 350, 420, 470, 520, 560] \\
&cap = [30, 10, 20, 30, 40, 20, 25, 30, 35, 20, 40, 25, 30, 50, 45]
\end{align*}
\]
where \( f_i \) and \( cap_i \) correspond to A1 through A15, respectively.

Distribution center demands:
\[
d = [30, 25, 20, 35, 25, 30, 25, 30]
\]
where \( d_j \) corresponds to B1 through B8, respectively.

Variable shipping cost matrix \( C = [c_{ij}] \) (rows: A1–A15, columns: B1–B8):

\[
C = \begin{bmatrix}
8 & 4 & 3 & 6 & 7 & 5 & 9 & 8 \\
5 & 2 & 3 & 5 & 6 & 4 & 7 & 6 \\
4 & 3 & 4 & 6 & 5 & 5 & 6 & 7 \\
9 & 7 & 5 & 8 & 9 & 6 & 10 & 7 \\
10 & 4 & 2 & 6 & 8 & 5 & 7 & 3 \\
6 & 5 & 4 & 5 & 7 & 6 & 8 & 5 \\
7 & 6 & 5 & 4 & 6 & 7 & 9 & 6 \\
5 & 4 & 6 & 3 & 5 & 6 & 7 & 6 \\
8 & 7 & 6 & 7 & 9 & 8 & 10 & 7 \\
6 & 5 & 7 & 4 & 6 & 5 & 7 & 5 \\
9 & 6 & 4 & 6 & 8 & 7 & 9 & 6 \\
7 & 5 & 6 & 5 & 6 & 5 & 8 & 5 \\
8 & 6 & 5 & 6 & 7 & 6 & 8 & 7 \\
9 & 5 & 3 & 5 & 7 & 4 & 6 & 4 \\
10 & 6 & 4 & 5 & 8 & 5 & 7 & 5 \\
\end{bmatrix}
\]

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if factory i is constructed, 0 otherwise.
- \( x_{ij} \geq 0 \): Amount shipped from factory i to distribution center j.

Objective:
Minimize total system cost (fixed + variable shipping):
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction at each distribution center:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]

2. Factory capacity and activation:
\[
\sum_{j \in J} x_{ij} \leq cap_i \cdot y_i \quad \forall i \in I
\]

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in I \\
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]

Summary of parameters (explicitly):

- \( I = \{A1, A2, ..., A15\} \)
- \( J = \{B1, B2, ..., B8\} \)
- \( f = [0, 175, 300, 375, 500, 200, 260, 220, 320, 280, 350, 420, 470, 520, 560] \)
- \( cap = [30, 10, 20, 30, 40, 20, 25, 30, 35, 20, 40, 25, 30, 50, 45] \)
- \( d = [30, 25, 20, 35, 25, 30, 25, 30] \)
- \( C \) as above.

This model finds the optimal subset of factories to construct and the optimal shipment plan from factories to distribution centers, minimizing the sum of fixed facility costs and variable shipping costs, while meeting all demand and respecting facility capacities.