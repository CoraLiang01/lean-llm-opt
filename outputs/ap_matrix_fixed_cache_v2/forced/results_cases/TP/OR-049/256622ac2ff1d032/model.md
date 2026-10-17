Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of shelves (with capacities $C_i$), and $J$ be the set of products (with values $v_j$ and weights $w_j$). Let $x_{ij} \in \mathbb{Z}_{\geq 0}$ denote the number of units of product $j$ placed on shelf $i$.

Maximize total value:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

Subject to, for each shelf $i$:
$$
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

and
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

Where:

Shelf capacities (in source order):

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

Products (in source order):

\[
\begin{array}{lll}
j & \text{ProductName} & v_j & w_j \\
1 & \text{Smartphone} & 200 & 1.0 \\
2 & \text{Laptop} & 1500 & 5.0 \\
3 & \text{Headphones} & 100 & 0.5 \\
4 & \text{Camera} & 800 & 2.0 \\
5 & \text{Smartwatch} & 250 & 0.3 \\
6 & \text{Tablet} & 600 & 1.5 \\
7 & \text{Bluetooth Speaker} & 150 & 1.0 \\
8 & \text{Keyboard} & 80 & 0.8 \\
9 & \text{Mouse} & 50 & 0.2 \\
10 & \text{Monitor} & 300 & 3.0 \\
11 & \text{Printer} & 400 & 4.0 \\
12 & \text{External Hard Drive} & 120 & 0.5 \\
13 & \text{Router} & 60 & 0.3 \\
14 & \text{Power Bank} & 40 & 0.4 \\
15 & \text{Memory Card} & 30 & 0.05 \\
16 & \text{USB Flash Drive} & 25 & 0.02 \\
17 & \text{Smart Home Hub} & 100 & 0.6 \\
18 & \text{Gaming Console} & 500 & 4.0 \\
19 & \text{Fitness Tracker} & 90 & 0.2 \\
20 & \text{E-Reader} & 180 & 0.5 \\
\end{array}
\]

All variables $x_{ij}$ are integer and nonnegative.

Retrieved Information:

Shelf capacities (source order):

- ShelfID 1: Capacity 5.0
- ShelfID 2: Capacity 7.0
- ShelfID 3: Capacity 6.0
- ShelfID 4: Capacity 8.0
- ShelfID 5: Capacity 5.5
- ShelfID 6: Capacity 9.0
- ShelfID 7: Capacity 6.5
- ShelfID 8: Capacity 7.5
- ShelfID 9: Capacity 8.2
- ShelfID 10: Capacity 5.7

Products (source order):

1. Smartphone: Value 200, Weight 1.0
2. Laptop: Value 1500, Weight 5.0
3. Headphones: Value 100, Weight 0.5
4. Camera: Value 800, Weight 2.0
5. Smartwatch: Value 250, Weight 0.3
6. Tablet: Value 600, Weight 1.5
7. Bluetooth Speaker: Value 150, Weight 1.0
8. Keyboard: Value 80, Weight 0.8
9. Mouse: Value 50, Weight 0.2
10. Monitor: Value 300, Weight 3.0
11. Printer: Value 400, Weight 4.0
12. External Hard Drive: Value 120, Weight 0.5
13. Router: Value 60, Weight 0.3
14. Power Bank: Value 40, Weight 0.4
15. Memory Card: Value 30, Weight 0.05
16. USB Flash Drive: Value 25, Weight 0.02
17. Smart Home Hub: Value 100, Weight 0.6
18. Gaming Console: Value 500, Weight 4.0
19. Fitness Tracker: Value 90, Weight 0.2
20. E-Reader: Value 180, Weight 0.5