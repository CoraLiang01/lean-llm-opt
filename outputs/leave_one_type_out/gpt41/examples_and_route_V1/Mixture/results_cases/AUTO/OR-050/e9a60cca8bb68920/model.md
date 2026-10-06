Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

**Indices:**
- $i$: ShelfID, $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- $j$: ProductName, $j \in \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$

**Parameters:**
- $c_i$: Capacity of shelf $i$ (from Capacity column)
- $v_j$: Value of product $j$ (from Value column)
- $w_j$: Weight of product $j$ (from Weight column)

**Data:**

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

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Smartphone            | 200   | 1.0    |
| Laptop                | 1500  | 5.0    |
| Headphones            | 100   | 0.5    |
| Camera                | 800   | 2.0    |
| Smartwatch            | 250   | 0.3    |
| Tablet                | 600   | 1.5    |
| Bluetooth Speaker     | 150   | 1.0    |
| Keyboard              | 80    | 0.8    |
| Mouse                 | 50    | 0.2    |
| Monitor               | 300   | 3.0    |
| Printer               | 400   | 4.0    |
| External Hard Drive   | 120   | 0.5    |
| Router                | 60    | 0.3    |
| Power Bank            | 40    | 0.4    |
| Memory Card           | 30    | 0.05   |
| USB Flash Drive       | 25    | 0.02   |
| Smart Home Hub        | 100   | 0.6    |
| Gaming Console        | 500   | 4.0    |
| Fitness Tracker       | 90    | 0.2    |
| E-Reader              | 180   | 0.5    |

---

### Mathematical Model

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all shelves $i$ and products $j$

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]
where $v_j$ is the value of product $j$ as given above.

**Constraints:**

1. **Shelf Capacity Constraints:**
   For each shelf $i$:
   \[
   \sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,2,\ldots,10\}
   \]
   where $w_j$ is the weight of product $j$ and $c_i$ is the capacity of shelf $i$.

2. **Minimum Allocation for First Product (Smartphone):**
   \[
   \sum_{i=1}^{10} x_{i,1} \geq 5
   \]
   where $x_{i,1}$ corresponds to the Smartphone.

3. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
   \]

---

**Explicitly, using the data:**

- For each shelf $i$ (ShelfID $1$ to $10$), $c_i$ is as in the table above.
- For each product $j$ (in the order listed), $v_j$ and $w_j$ are as in the table above.
- The minimum allocation constraint applies to the first product, Smartphone.

---

**Summary of Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10 \\
& \sum_{i=1}^{10} x_{i,1} \geq 5 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\end{align*}
\]

where all coefficients and identifiers are as given in the tables above.