**Sets and Indices:**

- Let $I$ be the set of displays (indexed by $i$), with ShelfID from capacity.csv: $I = \{1,2,3,4,5,6,7,8,9,10\}$.
- Let $J$ be the set of products (indexed by $j$), with ProductName from products.csv, in the given order:
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

**Parameters:**

- $c_i$ = Capacity of display $i$ (from capacity.csv)
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight of product $j$ (from products.csv)

**Decision Variables:**

- $x_{ij}$ = number of units of product $j$ placed on display $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

---

**Constraints:**

1. **Display Capacity Constraints** (for each display $i$):

   $$
   \sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
   $$

2. **Minimum Quantity of First Product (Smartphone) Across All Displays:**

   $$
   \sum_{i \in I} x_{i,1} \geq 5
   $$

3. **Nonnegativity and Integrality:**

   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   $$

---

**Parameter Data (in source order):**

*Displays (capacity.csv):*

| ShelfID | Capacity |
|---------|----------|
| 1       | 5        |
| 2       | 7        |
| 3       | 6        |
| 4       | 8        |
| 5       | 5.5      |
| 6       | 9        |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

*Products (products.csv):*

| $j$ | ProductName            | Value | Weight |
|-----|------------------------|-------|--------|
| 1   | Smartphone            | 200   | 1      |
| 2   | Laptop                | 1500  | 5      |
| 3   | Headphones            | 100   | 0.5    |
| 4   | Camera                | 800   | 2      |
| 5   | Smartwatch            | 250   | 0.3    |
| 6   | Tablet                | 600   | 1.5    |
| 7   | Bluetooth Speaker     | 150   | 1      |
| 8   | Keyboard              | 80    | 0.8    |
| 9   | Mouse                 | 50    | 0.2    |
| 10  | Monitor               | 300   | 3      |
| 11  | Printer               | 400   | 4      |
| 12  | External Hard Drive   | 120   | 0.5    |
| 13  | Router                | 60    | 0.3    |
| 14  | Power Bank            | 40    | 0.4    |
| 15  | Memory Card           | 30    | 0.05   |
| 16  | USB Flash Drive       | 25    | 0.02   |
| 17  | Smart Home Hub        | 100   | 0.6    |
| 18  | Gaming Console        | 500   | 4      |
| 19  | Fitness Tracker       | 90    | 0.2    |
| 20  | E-Reader              | 180   | 0.5    |

---

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10 \\
& \sum_{i=1}^{10} x_{i1} \geq 5 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
\end{align*}
$$

Where $c_i$, $v_j$, and $w_j$ are as given in the tables above, and $x_{ij}$ is the number of units of product $j$ placed on display $i$.