Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $i$ indexes shelves (using ShelfID from capacity.csv) and $j$ indexes products (using ProductName from products.csv). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $v_j$: Value of product $j$ (from Value in products.csv)
- $w_j$: Weight of product $j$ (from Weight in products.csv)
- $C_i$: Capacity of shelf $i$ (from Capacity in capacity.csv)

**Sets:**

- $I$: Set of shelves (ShelfID: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
- $J$: Set of products (ProductName: see below)

---

### Model

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

For each shelf $i \in I$:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i
\]

For all $i \in I$, $j \in J$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

### Parameter Tables

#### Shelves (from capacity.csv, in source order):

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

#### Products (from products.csv, in source order):

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

**Decision variables:**  
$x_{ij} \in \mathbb{Z}_{\geq 0}$ for all shelves $i$ and products $j$.

---

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j \in J} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in I \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I, j \in J
\end{align*}
\]

Where $I$ and $J$ and all parameters are as listed above.