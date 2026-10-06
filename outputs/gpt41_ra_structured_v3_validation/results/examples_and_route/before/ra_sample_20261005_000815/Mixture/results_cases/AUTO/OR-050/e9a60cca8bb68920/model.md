**Sets and Indices:**

- Let $i$ index ShelfID (displays): $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- Let $j$ index ProductName (products): $j \in \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$

**Parameters:**

- $C_i$ = Capacity of shelf $i$ (from "capacity.csv")
- $v_j$ = Value of product $j$ (from "products.csv")
- $w_j$ = Weight of product $j$ (from "products.csv")

**Decision Variables:**

- $x_{ij}$ = Number of units of product $j$ placed on shelf $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

---

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

where the products are indexed in the order given below.

---

**Constraints:**

1. **Shelf Capacity Constraints:**

For each shelf $i$:

$$
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
$$

2. **Minimum Allocation for First Product (Smartphone):**

$$
\sum_{i=1}^{10} x_{i,1} \geq 5
$$

(where $x_{i,1}$ corresponds to "Smartphone" as the first product in the list.)

3. **Nonnegativity and Integrality:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Data:**

- **Shelves (from "capacity.csv"):**

| ShelfID | Capacity |
|---------|----------|
| 1       | 5.0      |
| 2       | 7.0      |
| 3       | 6.0      |
| 4       | 8.0      |
| 5       | 5.5      |
| 6       | 9.0      |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

- **Products (from "products.csv"):**

| $j$ | ProductName           | $v_j$ | $w_j$  |
|-----|-----------------------|-------|--------|
| 1   | Smartphone           | 200   | 1.0    |
| 2   | Laptop               | 1500  | 5.0    |
| 3   | Headphones           | 100   | 0.5    |
| 4   | Camera               | 800   | 2.0    |
| 5   | Smartwatch           | 250   | 0.3    |
| 6   | Tablet               | 600   | 1.5    |
| 7   | Bluetooth Speaker    | 150   | 1.0    |
| 8   | Keyboard             | 80    | 0.8    |
| 9   | Mouse                | 50    | 0.2    |
| 10  | Monitor              | 300   | 3.0    |
| 11  | Printer              | 400   | 4.0    |
| 12  | External Hard Drive  | 120   | 0.5    |
| 13  | Router               | 60    | 0.3    |
| 14  | Power Bank           | 40    | 0.4    |
| 15  | Memory Card          | 30    | 0.05   |
| 16  | USB Flash Drive      | 25    | 0.02   |
| 17  | Smart Home Hub       | 100   | 0.6    |
| 18  | Gaming Console       | 500   | 4.0    |
| 19  | Fitness Tracker      | 90    | 0.2    |
| 20  | E-Reader             | 180   | 0.5    |

---

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& \sum_{i=1}^{10} x_{i,1} \geq 5 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
\end{align*}
$$

with $v_j$, $w_j$, and $C_i$ as given above, and $x_{ij}$ representing the number of units of product $j$ placed on shelf $i$.