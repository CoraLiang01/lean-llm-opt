Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of displays (ShelfID), and $J = \{1,2,\ldots,20\}$ be the set of products, indexed in the order given below.

Let $x_{ij} \geq 0$ (integer): number of units of product $j$ placed on display $i$.

Let $c_i$ be the capacity of display $i$.

Let $v_j$ be the value of product $j$.

Let $w_j$ be the weight of product $j$.

Let product 1 be "Smartphone".

Maximize total value:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

Subject to:

1. Capacity constraints for each display:
$$
\sum_{j=1}^{20} w_j x_{ij} \leq c_i \quad \forall i=1,\ldots,10
$$

2. Minimum total quantity of the first product ("Smartphone"):
$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

3. Non-negativity and integrality:
$$
x_{ij} \geq 0,\quad x_{ij} \in \mathbb{Z} \quad \forall i=1,\ldots,10;\ j=1,\ldots,20
$$

Where:

Display capacities (in order):

\[
\begin{align*}
c_1 &= 5.0 \\
c_2 &= 7.0 \\
c_3 &= 6.0 \\
c_4 &= 8.0 \\
c_5 &= 5.5 \\
c_6 &= 9.0 \\
c_7 &= 6.5 \\
c_8 &= 7.5 \\
c_9 &= 8.2 \\
c_{10} &= 5.7 \\
\end{align*}
\]

Products (in order):

\[
\begin{array}{lll}
j & \text{ProductName} & (v_j, w_j) \\
1 & \text{Smartphone} & (200,\ 1.0) \\
2 & \text{Laptop} & (1500,\ 5.0) \\
3 & \text{Headphones} & (100,\ 0.5) \\
4 & \text{Camera} & (800,\ 2.0) \\
5 & \text{Smartwatch} & (250,\ 0.3) \\
6 & \text{Tablet} & (600,\ 1.5) \\
7 & \text{Bluetooth Speaker} & (150,\ 1.0) \\
8 & \text{Keyboard} & (80,\ 0.8) \\
9 & \text{Mouse} & (50,\ 0.2) \\
10 & \text{Monitor} & (300,\ 3.0) \\
11 & \text{Printer} & (400,\ 4.0) \\
12 & \text{External Hard Drive} & (120,\ 0.5) \\
13 & \text{Router} & (60,\ 0.3) \\
14 & \text{Power Bank} & (40,\ 0.4) \\
15 & \text{Memory Card} & (30,\ 0.05) \\
16 & \text{USB Flash Drive} & (25,\ 0.02) \\
17 & \text{Smart Home Hub} & (100,\ 0.6) \\
18 & \text{Gaming Console} & (500,\ 4.0) \\
19 & \text{Fitness Tracker} & (90,\ 0.2) \\
20 & \text{E-Reader} & (180,\ 0.5) \\
\end{array}
\]

All coefficients and identifiers are as retrieved and in source order.