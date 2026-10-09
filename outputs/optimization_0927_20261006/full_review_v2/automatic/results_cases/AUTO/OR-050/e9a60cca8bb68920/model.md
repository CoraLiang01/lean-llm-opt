Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i \in \{1,2,\ldots,10\}$ and $j \in \{1,2,\ldots,20\}$, corresponding to the order of shelves and products as given below.

**Product Indexing (in order from products.csv):**
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

**Shelf Indexing (in order from capacity.csv):**
1. ShelfID 1 (Capacity 5.0)
2. ShelfID 2 (Capacity 7.0)
3. ShelfID 3 (Capacity 6.0)
4. ShelfID 4 (Capacity 8.0)
5. ShelfID 5 (Capacity 5.5)
6. ShelfID 6 (Capacity 9.0)
7. ShelfID 7 (Capacity 6.5)
8. ShelfID 8 (Capacity 7.5)
9. ShelfID 9 (Capacity 8.2)
10. ShelfID 10 (Capacity 5.7)

**Parameters:**

Let $v_j$ be the value of product $j$:

\[
\begin{align*}
v_1 &= 200 \\
v_2 &= 1500 \\
v_3 &= 100 \\
v_4 &= 800 \\
v_5 &= 250 \\
v_6 &= 600 \\
v_7 &= 150 \\
v_8 &= 80 \\
v_9 &= 50 \\
v_{10} &= 300 \\
v_{11} &= 400 \\
v_{12} &= 120 \\
v_{13} &= 60 \\
v_{14} &= 40 \\
v_{15} &= 30 \\
v_{16} &= 25 \\
v_{17} &= 100 \\
v_{18} &= 500 \\
v_{19} &= 90 \\
v_{20} &= 180 \\
\end{align*}
\]

Let $w_j$ be the weight of product $j$:

\[
\begin{align*}
w_1 &= 1.0 \\
w_2 &= 5.0 \\
w_3 &= 0.5 \\
w_4 &= 2.0 \\
w_5 &= 0.3 \\
w_6 &= 1.5 \\
w_7 &= 1.0 \\
w_8 &= 0.8 \\
w_9 &= 0.2 \\
w_{10} &= 3.0 \\
w_{11} &= 4.0 \\
w_{12} &= 0.5 \\
w_{13} &= 0.3 \\
w_{14} &= 0.4 \\
w_{15} &= 0.05 \\
w_{16} &= 0.02 \\
w_{17} &= 0.6 \\
w_{18} &= 4.0 \\
w_{19} &= 0.2 \\
w_{20} &= 0.5 \\
\end{align*}
\]

Let $C_i$ be the capacity of shelf $i$:

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
   \sum_{j=1}^{20} w_j x_{ij} &\leq C_i
   \end{align*}

2. **Minimum Allocation of First Product (Smartphone):**
   \[
   \sum_{i=1}^{10} x_{i1} \geq 5
   \]

3. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\ j = 1,\ldots,20
   \]

---

**All coefficients and identifiers are as retrieved and in original order.**