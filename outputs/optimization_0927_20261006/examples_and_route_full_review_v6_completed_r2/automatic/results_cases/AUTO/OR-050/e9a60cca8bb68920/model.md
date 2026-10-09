Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Let displays (shelves) be indexed by $i \in \{1,2,3,4,5,6,7,8,9,10\}$, corresponding to ShelfID in the order given.
Let products be indexed by $j \in \{1,2,\ldots,20\}$, corresponding to the ProductName order below.

Define:
- $c_i$ = capacity of shelf $i$ (from "Capacity" column)
- $v_j$ = value of product $j$ (from "Value" column)
- $w_j$ = weight of product $j$ (from "Weight" column)

Product order (for $j$):
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

Shelf order (for $i$):
1. ShelfID 1 ($c_1 = 5.0$)
2. ShelfID 2 ($c_2 = 7.0$)
3. ShelfID 3 ($c_3 = 6.0$)
4. ShelfID 4 ($c_4 = 8.0$)
5. ShelfID 5 ($c_5 = 5.5$)
6. ShelfID 6 ($c_6 = 9.0$)
7. ShelfID 7 ($c_7 = 6.5$)
8. ShelfID 8 ($c_8 = 7.5$)
9. ShelfID 9 ($c_9 = 8.2$)
10. ShelfID 10 ($c_{10} = 5.7$)

Product values and weights:
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

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]

Subject to:

1. Shelf capacity constraints (for each shelf $i$):
\[
\sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10
\]
That is,
\begin{align*}
\sum_{j=1}^{20} w_j x_{1j} &\leq 5.0 \\
\sum_{j=1}^{20} w_j x_{2j} &\leq 7.0 \\
\sum_{j=1}^{20} w_j x_{3j} &\leq 6.0 \\
\sum_{j=1}^{20} w_j x_{4j} &\leq 8.0 \\
\sum_{j=1}^{20} w_j x_{5j} &\leq 5.5 \\
\sum_{j=1}^{20} w_j x_{6j} &\leq 9.0 \\
\sum_{j=1}^{20} w_j x_{7j} &\leq 6.5 \\
\sum_{j=1}^{20} w_j x_{8j} &\leq 7.5 \\
\sum_{j=1}^{20} w_j x_{9j} &\leq 8.2 \\
\sum_{j=1}^{20} w_j x_{10j} &\leq 5.7 \\
\end{align*}

2. Minimum total quantity of the first product (Smartphone) across all shelves:
\[
\sum_{i=1}^{10} x_{i1} \geq 5
\]

3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
\]

Where:
- $x_{ij}$: number of units of product $j$ placed on shelf $i$
- $v_j$: value of product $j$ (see table above)
- $w_j$: weight of product $j$ (see table above)
- $c_i$: capacity of shelf $i$ (see list above)

All data and indices are as retrieved and ordered above.