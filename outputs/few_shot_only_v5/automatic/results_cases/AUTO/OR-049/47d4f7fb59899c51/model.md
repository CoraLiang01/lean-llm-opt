Let $x_{ij}$ be the number of units of product $j$ placed on shelf $i$, where $i$ indexes shelves and $j$ indexes products. All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $S$ be the set of shelves, with ShelfID and Capacity as below:

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

- Let $P$ be the set of products, with ProductName, Value, and Weight as below:

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

**Model:**

**Objective:**
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of product $j$.

**Subject to:**

1. **Shelf Capacity Constraints:**
   \[
   \sum_{j \in P} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in S
   \]
   where $w_j$ is the Weight of product $j$, and $c_i$ is the Capacity of shelf $i$.

2. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
   \]

**Decision Variables:**

- $x_{ij}$: Number of units of product $j$ to place on shelf $i$ (integer, $\geq 0$).

**All identifiers and coefficients are as given in the tables above.**