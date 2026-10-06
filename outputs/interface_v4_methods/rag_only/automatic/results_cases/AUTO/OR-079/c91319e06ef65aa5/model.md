Let:
- \( I = \{A1, A2, ..., A15\} \) be the set of potential factory sites.
- \( J = \{B1, B2, ..., B8\} \) be the set of distribution centers.

Parameters:
- Fixed costs for each factory \( i \in I \):
  \[
  f = [0, 175, 300, 375, 500, 200, 260, 220, 320, 280, 350, 420, 470, 520, 560]
  \]
  where \( f_i \) is the fixed cost of opening factory \( i \), in the order A1–A15.

- Capacity for each factory \( i \in I \):
  \[
  cap = [30, 10, 20, 30, 40, 20, 25, 30, 35, 20, 40, 25, 30, 50, 45]
  \]
  where \( cap_i \) is the capacity of factory \( i \), in the order A1–A15.

- Demand at each distribution center \( j \in J \):
  \[
  d = [30, 25, 20, 35, 25, 30, 25, 30]
  \]
  where \( d_j \) is the demand at distribution center \( j \), in the order B1–B8.

- Variable shipping costs from each factory \( i \) to each distribution center \( j \):
  \[
  c = \begin{bmatrix}
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
  where \( c_{ij} \) is the shipping cost from factory \( i \) to distribution center \( j \), with rows ordered A1–A15 and columns B1–B8.

Decision Variables:
- \( y_i \in \{0,1\} \): 1 if factory \( i \) is constructed, 0 otherwise.
- \( x_{ij} \geq 0 \): quantity shipped from factory \( i \) to distribution center \( j \).

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

3. Non-negativity and binary constraints:
   \[
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

Summary of parameters (explicit vectors/matrix):

- \( f = [0, 175, 300, 375, 500, 200, 260, 220, 320, 280, 350, 420, 470, 520, 560] \)
- \( cap = [30, 10, 20, 30, 40, 20, 25, 30, 35, 20, 40, 25, 30, 50, 45] \)
- \( d = [30, 25, 20, 35, 25, 30, 25, 30] \)
- \( c \) as the 15x8 matrix above.

This model finds the optimal subset of factories to construct and the optimal shipment plan to minimize total system cost while meeting all demand and respecting facility capacities.