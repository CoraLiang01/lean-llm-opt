Let $x_{ij}$ be the number of units of product $j$ (ProductName) to be placed on shelf $i$ (ShelfID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $S$ = set of shelves, indexed by $i$ (ShelfID: 1, 2, ..., 10)
- $P$ = set of products, indexed by $j$ (ProductName: Smartphone, Laptop, ..., E-Reader)
- $v_j$ = Value of product $j$ (from "Value" column)
- $w_j$ = Weight of product $j$ (from "Weight" column)
- $C_i$ = Capacity of shelf $i$ (from "Capacity" column)

**Objective:**

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

**Subject to:**

For each shelf $i \in S$:
$$
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
$$

For all $i \in S$, $j \in P$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Data:**

**Shelves (from capacity.csv):**

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

**Products (from products.csv):**

| ProductName            | Value | Weight |
|------------------------|-------|--------|
| Smartphone             | 200   | 1.0    |
| Laptop                 | 1500  | 5.0    |
| Headphones             | 100   | 0.5    |
| Camera                 | 800   | 2.0    |
| Smartwatch             | 250   | 0.3    |
| Tablet                 | 600   | 1.5    |
| Bluetooth Speaker      | 150   | 1.0    |
| Keyboard               | 80    | 0.8    |
| Mouse                  | 50    | 0.2    |
| Monitor                | 300   | 3.0    |
| Printer                | 400   | 4.0    |
| External Hard Drive    | 120   | 0.5    |
| Router                 | 60    | 0.3    |
| Power Bank             | 40    | 0.4    |
| Memory Card            | 30    | 0.05   |
| USB Flash Drive        | 25    | 0.02   |
| Smart Home Hub         | 100   | 0.6    |
| Gaming Console         | 500   | 4.0    |
| Fitness Tracker        | 90    | 0.2    |
| E-Reader               | 180   | 0.5    |

**Decision variables:**

$x_{ij}$: Number of units of product $j$ to be placed on shelf $i$, for all $i$ (ShelfID 1–10), $j$ (all ProductName above), $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i = 1, \ldots, 10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10,\ j = 1, \ldots, 20
\end{align*}
$$

Where $v_j$, $w_j$, $C_i$ are as listed above, and product/shelf indices correspond to the original ProductName and ShelfID.