Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

Let $S$ be the set of shelves, indexed by ShelfID:
$$
S = \{1,2,3,4,5,6,7,8,9,10\}
$$

Let $P$ be the set of products, indexed by their ProductName (in the order returned):

\[
\begin{array}{ll}
1: & \text{Smartphone} \\
2: & \text{Laptop} \\
3: & \text{Headphones} \\
4: & \text{Camera} \\
5: & \text{Smartwatch} \\
6: & \text{Tablet} \\
7: & \text{Bluetooth Speaker} \\
8: & \text{Keyboard} \\
9: & \text{Mouse} \\
10: & \text{Monitor} \\
11: & \text{Printer} \\
12: & \text{External Hard Drive} \\
13: & \text{Router} \\
14: & \text{Power Bank} \\
15: & \text{Memory Card} \\
16: & \text{USB Flash Drive} \\
17: & \text{Smart Home Hub} \\
18: & \text{Gaming Console} \\
19: & \text{Fitness Tracker} \\
20: & \text{E-Reader} \\
\end{array}
\]

Let $v_j$ be the value of product $j$ and $w_j$ its weight, as follows (in the same order):

\[
\begin{array}{lll}
\text{ProductName} & v_j & w_j \\
\hline
\text{Smartphone} & 200 & 1 \\
\text{Laptop} & 1500 & 5 \\
\text{Headphones} & 100 & 0.5 \\
\text{Camera} & 800 & 2 \\
\text{Smartwatch} & 250 & 0.3 \\
\text{Tablet} & 600 & 1.5 \\
\text{Bluetooth Speaker} & 150 & 1 \\
\text{Keyboard} & 80 & 0.8 \\
\text{Mouse} & 50 & 0.2 \\
\text{Monitor} & 300 & 3 \\
\text{Printer} & 400 & 4 \\
\text{External Hard Drive} & 120 & 0.5 \\
\text{Router} & 60 & 0.3 \\
\text{Power Bank} & 40 & 0.4 \\
\text{Memory Card} & 30 & 0.05 \\
\text{USB Flash Drive} & 25 & 0.02 \\
\text{Smart Home Hub} & 100 & 0.6 \\
\text{Gaming Console} & 500 & 4 \\
\text{Fitness Tracker} & 90 & 0.2 \\
\text{E-Reader} & 180 & 0.5 \\
\end{array}
\]

Let $C_i$ be the capacity of shelf $i$ (from the "Capacity" column):

\[
\begin{array}{ll}
\text{ShelfID} & C_i \\
1 & 5 \\
2 & 7 \\
3 & 6 \\
4 & 8 \\
5 & 5.5 \\
6 & 9 \\
7 & 6.5 \\
8 & 7.5 \\
9 & 8.2 \\
10 & 5.7 \\
\end{array}
\]

The mathematical model is:

---

**Variables:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
\]

---

**Objective:**

\[
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
\]

---

**Constraints:**

1. **Shelf capacity constraints:**

\[
\sum_{j \in P} w_j \, x_{ij} \leq C_i \qquad \forall i \in S
\]

That is, for each shelf $i$:

\[
\begin{align*}
\sum_{j=1}^{20} w_j x_{i j} &\leq C_i
\end{align*}
\]

with $w_j$ and $C_i$ as above.

2. **Minimum allocation of the first product (Smartphone) across all shelves:**

\[
\sum_{i \in S} x_{i,1} \geq 5
\]

where $x_{i,1}$ is the number of Smartphones on shelf $i$.

3. **Nonnegativity and integrality:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S, \forall j \in P
\]

---

**All parameters are as listed above, with all identifiers and coefficients preserved in source order.**