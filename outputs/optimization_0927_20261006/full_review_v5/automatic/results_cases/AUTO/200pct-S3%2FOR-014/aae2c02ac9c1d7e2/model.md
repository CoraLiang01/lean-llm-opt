**Sets and Indices:**

- Let $I$ be the set of products, indexed by $i$ (ProductName from products.csv).
- Let $J$ be the set of shelves, indexed by $j$ (ShelfID from capacity.csv).

**Parameters:**

- $v_i$: Value of one unit of product $i$ (Value column).
- $w_i$: Weight of one unit of product $i$ (Weight column).
- $C_j$: Capacity of shelf $j$ (Capacity column).

**Decision Variables:**

- $x_{ij}$: Number of units of product $i$ to place on shelf $j$, $x_{ij} \in \mathbb{Z}_{\geq 0}$.

---

### Mathematical Model

**Objective:**

$$
\max \sum_{j \in J} \sum_{i \in I} v_i \cdot x_{ij}
$$

**Subject to:**

For each shelf $j \in J$:
$$
\sum_{i \in I} w_i \cdot x_{ij} \leq C_j
$$

For all $i \in I$, $j \in J$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

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

**Summary of Model:**

- Maximize total value of products allocated to shelves.
- For each shelf, total weight of allocated products cannot exceed its capacity.
- All allocations are nonnegative integers.