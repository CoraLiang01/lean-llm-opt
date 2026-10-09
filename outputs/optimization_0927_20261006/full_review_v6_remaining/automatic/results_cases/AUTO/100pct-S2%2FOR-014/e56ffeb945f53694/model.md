**Sets and Indices:**

- Let $S$ be the set of shelves, indexed by $i$ (with ShelfID from capacity.csv).
- Let $P$ be the set of products, indexed by $j$ (with ProductName from products.csv).

**Parameters:**

- $v_j$: Value of one unit of product $j$ (from Value in products.csv).
- $w_j$: Weight of one unit of product $j$ (from Weight in products.csv).
- $C_i$: Capacity of shelf $i$ (from Capacity in capacity.csv).

**Decision Variables:**

- $x_{ij}$: Number of units of product $j$ to place on shelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

---

**Objective:**

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

---

**Constraints:**

For each shelf $i \in S$:
$$
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
$$

For all $i \in S$, $j \in P$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Data (in source order):**

*Shelves (capacity.csv):*

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

*Products (products.csv):*

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

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i \in \{1,\ldots,10\}} \sum_{j \in P} v_j x_{ij} \\[2ex]
\text{s.t.} \quad & \sum_{j \in P} w_j x_{ij} \leq C_i, \quad \forall i \in \{1,\ldots,10\} \\[2ex]
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,\ldots,10\},\ j \in P
\end{align*}
$$

Where $P$ and all coefficients are as listed above.