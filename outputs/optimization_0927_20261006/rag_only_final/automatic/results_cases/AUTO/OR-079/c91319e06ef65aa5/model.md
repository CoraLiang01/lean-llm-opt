Let:
- \( I = \{1,2,\ldots,15\} \) index the potential factory sites (A1–A15).
- \( J = \{1,2,\ldots,8\} \) index the distribution centers (B1–B8).

Parameters:
- Fixed facility opening costs (in the order A1–A15):  
  \( f = [0, 175, 300, 375, 500, 200, 260, 220, 320, 280, 350, 420, 470, 520, 560] \)
- Facility capacities (in the order A1–A15):  
  \( \text{cap} = [30, 10, 20, 30, 40, 20, 25, 30, 35, 20, 40, 25, 30, 50, 45] \)
- Demand at each distribution center (in the order B1–B8):  
  \( d = [30, 25, 20, 35, 25, 30, 25, 30] \)
- Shipping costs from each factory to each distribution center (\( c_{ij} \)), as a 15×8 matrix (rows: A1–A15, columns: B1–B8):

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

Decision variables:
- \( y_i \in \{0,1\} \): 1 if factory \( i \) is constructed, 0 otherwise.
- \( x_{ij} \geq 0 \): quantity shipped from factory \( i \) to distribution center \( j \).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^{15} f_i y_i + \sum_{i=1}^{15} \sum_{j=1}^{8} c_{ij} x_{ij} \\
\text{subject to} \quad
& \sum_{i=1}^{15} x_{ij} = d_j \quad \forall j = 1,\ldots,8 \quad \text{(meet all demand)} \\
& \sum_{j=1}^{8} x_{ij} \leq \text{cap}_i \cdot y_i \quad \forall i = 1,\ldots,15 \quad \text{(respect capacity if open)} \\
& y_i \in \{0,1\} \quad \forall i = 1,\ldots,15 \\
& x_{ij} \geq 0 \quad \forall i = 1,\ldots,15, \; j = 1,\ldots,8
\end{align*}
\]

Where:
- \( f_i \) is the fixed cost for factory \( i \) (see vector above).
- \( c_{ij} \) is the shipping cost from factory \( i \) to distribution center \( j \) (see matrix above).
- \( \text{cap}_i \) is the capacity of factory \( i \) (see vector above).
- \( d_j \) is the demand at distribution center \( j \) (see vector above).

This model minimizes the total system cost (fixed + variable shipping), chooses which factories to open, and determines optimal shipment flows to meet all demand.