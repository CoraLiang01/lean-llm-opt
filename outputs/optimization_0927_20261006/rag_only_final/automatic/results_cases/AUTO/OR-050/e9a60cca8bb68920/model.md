Let:
- \( I = \{1,2,3,4,5,6,7,8,9,10\} \) be the set of displays, indexed by ShelfID as given in capacity.csv.
- \( J = \{1,2,\ldots,20\} \) be the set of products, indexed in the order of products.csv:
  1. Smartphone
  2. Laptop
  3. Headphones
  4. Camera
  5. Smartwatch
  6. Tablet
  7. Bluetooth Speaker
  8. Keyboard
  9. Mouse
  10. Monitor
  11. Printer
  12. External Hard Drive
  13. Router
  14. Power Bank
  15. Memory Card
  16. USB Flash Drive
  17. Smart Home Hub
  18. Gaming Console
  19. Fitness Tracker
  20. E-Reader

Parameters:
- \( C_i \): Capacity of display \( i \) (from capacity.csv)
- \( v_j \): Value of product \( j \) (from products.csv)
- \( w_j \): Weight of product \( j \) (from products.csv)

Data:
From capacity.csv:
\[
\begin{array}{ll}
\text{ShelfID} & \text{Capacity} \\
1 & 5.0 \\
2 & 7.0 \\
3 & 6.0 \\
4 & 8.0 \\
5 & 5.5 \\
6 & 9.0 \\
7 & 6.5 \\
8 & 7.5 \\
9 & 8.2 \\
10 & 5.7 \\
\end{array}
\]

From products.csv (in order):
\[
\begin{array}{lll}
j & \text{ProductName} & (v_j, w_j) \\
1 & \text{Smartphone} & (200, 1.0) \\
2 & \text{Laptop} & (1500, 5.0) \\
3 & \text{Headphones} & (100, 0.5) \\
4 & \text{Camera} & (800, 2.0) \\
5 & \text{Smartwatch} & (250, 0.3) \\
6 & \text{Tablet} & (600, 1.5) \\
7 & \text{Bluetooth Speaker} & (150, 1.0) \\
8 & \text{Keyboard} & (80, 0.8) \\
9 & \text{Mouse} & (50, 0.2) \\
10 & \text{Monitor} & (300, 3.0) \\
11 & \text{Printer} & (400, 4.0) \\
12 & \text{External Hard Drive} & (120, 0.5) \\
13 & \text{Router} & (60, 0.3) \\
14 & \text{Power Bank} & (40, 0.4) \\
15 & \text{Memory Card} & (30, 0.05) \\
16 & \text{USB Flash Drive} & (25, 0.02) \\
17 & \text{Smart Home Hub} & (100, 0.6) \\
18 & \text{Gaming Console} & (500, 4.0) \\
19 & \text{Fitness Tracker} & (90, 0.2) \\
20 & \text{E-Reader} & (180, 0.5) \\
\end{array}
\]

Decision variables:
- \( x_{ij} \): Number of units of product \( j \) placed on display \( i \), for \( i \in I, j \in J \).
- Domain: \( x_{ij} \in \mathbb{Z}_+ \) (nonnegative integers).

Model:

\[
\begin{align*}
\text{Maximize} \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{subject to} \quad
& \sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i = 1,\ldots,10 \\
& \sum_{i=1}^{10} x_{i1} \geq 5 \\
& x_{ij} \in \mathbb{Z}_+, \quad \forall i = 1,\ldots,10, \; j = 1,\ldots,20
\end{align*}
\]

Where:
- \( v_j \) and \( w_j \) are as listed above for each product \( j \).
- \( C_i \) is the capacity for each display \( i \) as listed above.

Explicitly, the constraints are:

For each display \( i \):
\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i
\]
with the following values for \( C_i \):

\[
\begin{align*}
C_1 &= 5.0 \\
C_2 &= 7.0 \\
C_3 &= 6.0 \\
C_4 &= 8.0 \\
C_5 &= 5.5 \\
C_6 &= 9.0 \\
C_7 &= 6.5 \\
C_8 &= 7.5 \\
C_9 &= 8.2 \\
C_{10} &= 5.7 \\
\end{align*}
\]

And the minimum total quantity constraint for the first product (Smartphone):

\[
\sum_{i=1}^{10} x_{i1} \geq 5
\]

All variables \( x_{ij} \) are nonnegative integers.

This is a complete numerical mixed-integer linear programming formulation for the described retail product allocation problem.