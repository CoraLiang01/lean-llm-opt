Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, for $i \in \{1,2,\ldots,10\}$ and $j \in \{1,2,\ldots,20\}$, corresponding to the order of ShelfID and ProductName as given below.

**Product Indexing (in source order):**
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

**Shelf Indexing (in source order):**
1. ShelfID 1 (Capacity: 5)
2. ShelfID 2 (Capacity: 7)
3. ShelfID 3 (Capacity: 6)
4. ShelfID 4 (Capacity: 8)
5. ShelfID 5 (Capacity: 5.5)
6. ShelfID 6 (Capacity: 9)
7. ShelfID 7 (Capacity: 6.5)
8. ShelfID 8 (Capacity: 7.5)
9. ShelfID 9 (Capacity: 8.2)
10. ShelfID 10 (Capacity: 5.7)

**Parameters:**

Let $v_j$ be the value of product $j$ and $w_j$ be the weight of product $j$:

\[
\begin{array}{lll}
j & \text{ProductName} & v_j & w_j \\
1 & \text{Smartphone} & 200 & 1 \\
2 & \text{Laptop} & 1500 & 5 \\
3 & \text{Headphones} & 100 & 0.5 \\
4 & \text{Camera} & 800 & 2 \\
5 & \text{Smartwatch} & 250 & 0.3 \\
6 & \text{Tablet} & 600 & 1.5 \\
7 & \text{Bluetooth Speaker} & 150 & 1 \\
8 & \text{Keyboard} & 80 & 0.8 \\
9 & \text{Mouse} & 50 & 0.2 \\
10 & \text{Monitor} & 300 & 3 \\
11 & \text{Printer} & 400 & 4 \\
12 & \text{External Hard Drive} & 120 & 0.5 \\
13 & \text{Router} & 60 & 0.3 \\
14 & \text{Power Bank} & 40 & 0.4 \\
15 & \text{Memory Card} & 30 & 0.05 \\
16 & \text{USB Flash Drive} & 25 & 0.02 \\
17 & \text{Smart Home Hub} & 100 & 0.6 \\
18 & \text{Gaming Console} & 500 & 4 \\
19 & \text{Fitness Tracker} & 90 & 0.2 \\
20 & \text{E-Reader} & 180 & 0.5 \\
\end{array}
\]

Let $C_i$ be the capacity of shelf $i$:

\[
\begin{array}{ll}
i & C_i \\
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

---

### Mathematical Model

**Decision Variables:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]

**Subject to:**

1. **Shelf Capacity Constraints:**
   \[
   \sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
   \]
   That is, for each shelf $i$:
   \begin{align*}
   \sum_{j=1}^{20} w_j x_{1j} &\leq 5 \\
   \sum_{j=1}^{20} w_j x_{2j} &\leq 7 \\
   \sum_{j=1}^{20} w_j x_{3j} &\leq 6 \\
   \sum_{j=1}^{20} w_j x_{4j} &\leq 8 \\
   \sum_{j=1}^{20} w_j x_{5j} &\leq 5.5 \\
   \sum_{j=1}^{20} w_j x_{6j} &\leq 9 \\
   \sum_{j=1}^{20} w_j x_{7j} &\leq 6.5 \\
   \sum_{j=1}^{20} w_j x_{8j} &\leq 7.5 \\
   \sum_{j=1}^{20} w_j x_{9j} &\leq 8.2 \\
   \sum_{j=1}^{20} w_j x_{10j} &\leq 5.7 \\
   \end{align*}

2. **Minimum Quantity of First Product (Smartphone):**
   \[
   \sum_{i=1}^{10} x_{i1} \geq 5
   \]

3. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
   \]

---

**All coefficients and identifiers are as retrieved and in original order.**