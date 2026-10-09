Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, for $i \in \{1,2,\ldots,10\}$ and $j \in \{1,2,\ldots,20\}$, corresponding to the order of shelves and products as given below.

**Shelves (from capacity.csv, in order):**

| $i$ | ShelfID | Capacity |
|-----|---------|----------|
| 1   | 1       | 5        |
| 2   | 2       | 7        |
| 3   | 3       | 6        |
| 4   | 4       | 8        |
| 5   | 5       | 5.5      |
| 6   | 6       | 9        |
| 7   | 7       | 6.5      |
| 8   | 8       | 7.5      |
| 9   | 9       | 8.2      |
| 10  | 10      | 5.7      |

**Products (from products.csv, in order):**

| $j$ | ProductName             | Value | Weight |
|-----|------------------------|-------|--------|
| 1   | Smartphone             | 200   | 1      |
| 2   | Laptop                 | 1500  | 5      |
| 3   | Headphones             | 100   | 0.5    |
| 4   | Camera                 | 800   | 2      |
| 5   | Smartwatch             | 250   | 0.3    |
| 6   | Tablet                 | 600   | 1.5    |
| 7   | Bluetooth Speaker      | 150   | 1      |
| 8   | Keyboard               | 80    | 0.8    |
| 9   | Mouse                  | 50    | 0.2    |
| 10  | Monitor                | 300   | 3      |
| 11  | Printer                | 400   | 4      |
| 12  | External Hard Drive    | 120   | 0.5    |
| 13  | Router                 | 60    | 0.3    |
| 14  | Power Bank             | 40    | 0.4    |
| 15  | Memory Card            | 30    | 0.05   |
| 16  | USB Flash Drive        | 25    | 0.02   |
| 17  | Smart Home Hub         | 100   | 0.6    |
| 18  | Gaming Console         | 500   | 4      |
| 19  | Fitness Tracker        | 90    | 0.2    |
| 20  | E-Reader               | 180   | 0.5    |

---

### Mathematical Model

**Decision Variables:**

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on shelf $i$.

---

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}
$$

where $v_j$ is the value of product $j$ (see table above).

---

**Constraints:**

1. **Shelf Capacity Constraints (for each shelf $i$):**

For each $i = 1, \ldots, 10$,

$$
\sum_{j=1}^{20} w_j \, x_{ij} \leq c_i
$$

where $w_j$ is the weight of product $j$ and $c_i$ is the capacity of shelf $i$ (see tables above).

2. **Minimum Total Quantity of First Product:**

$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

3. **Nonnegativity and Integrality:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
$$

---

**Parameter Values (in source order):**

- Shelf capacities $c_i$ (for $i=1$ to $10$): 5, 7, 6, 8, 5.5, 9, 6.5, 7.5, 8.2, 5.7
- Product values $v_j$ (for $j=1$ to $20$): 200, 1500, 100, 800, 250, 600, 150, 80, 50, 300, 400, 120, 60, 40, 30, 25, 100, 500, 90, 180
- Product weights $w_j$ (for $j=1$ to $20$): 1, 5, 0.5, 2, 0.3, 1.5, 1, 0.8, 0.2, 3, 4, 0.5, 0.3, 0.4, 0.05, 0.02, 0.6, 4, 0.2, 0.5

---

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij} \\[2ex]
\text{s.t.} \quad
& \sum_{j=1}^{20} w_j \, x_{i j} \leq c_i \quad && \forall i = 1,\ldots,10 \\[2ex]
& \sum_{i=1}^{10} x_{i1} \geq 5 \\[2ex]
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad && \forall i = 1,\ldots,10;\; j = 1,\ldots,20
\end{align*}
$$

with all coefficients and identifiers as listed above.