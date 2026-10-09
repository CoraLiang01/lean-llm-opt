Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all shelves $i$ and products $j$.

Let $S$ be the set of shelves, indexed by ShelfID:
$$
S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}
$$

Let $P$ be the set of products, indexed by ProductName:
$$
P = \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}
$$

Let $v_j$ be the value of product $j$ and $w_j$ be the weight of product $j$ (from products.csv).
Let $C_i$ be the capacity of shelf $i$ (from capacity.csv).

#### Objective:
Maximize the total value of products allocated to all shelves:
$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

#### Subject to:

For each shelf $i \in S$ (using the original order and identifiers):

- Shelf 1: $\sum_{j \in P} w_j \cdot x_{1j} \leq 5.0$
- Shelf 2: $\sum_{j \in P} w_j \cdot x_{2j} \leq 7.0$
- Shelf 3: $\sum_{j \in P} w_j \cdot x_{3j} \leq 6.0$
- Shelf 4: $\sum_{j \in P} w_j \cdot x_{4j} \leq 8.0$
- Shelf 5: $\sum_{j \in P} w_j \cdot x_{5j} \leq 5.5$
- Shelf 6: $\sum_{j \in P} w_j \cdot x_{6j} \leq 9.0$
- Shelf 7: $\sum_{j \in P} w_j \cdot x_{7j} \leq 6.5$
- Shelf 8: $\sum_{j \in P} w_j \cdot x_{8j} \leq 7.5$
- Shelf 9: $\sum_{j \in P} w_j \cdot x_{9j} \leq 8.2$
- Shelf 10: $\sum_{j \in P} w_j \cdot x_{10j} \leq 5.7$

For all $i \in S$, $j \in P$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

#### Data

**Shelves (from capacity.csv, in source order):**

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

**Products (from products.csv, in source order):**

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

#### Complete Model

$$
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j x_{1j} \leq 5.0 \\
                  & \sum_{j=1}^{20} w_j x_{2j} \leq 7.0 \\
                  & \sum_{j=1}^{20} w_j x_{3j} \leq 6.0 \\
                  & \sum_{j=1}^{20} w_j x_{4j} \leq 8.0 \\
                  & \sum_{j=1}^{20} w_j x_{5j} \leq 5.5 \\
                  & \sum_{j=1}^{20} w_j x_{6j} \leq 9.0 \\
                  & \sum_{j=1}^{20} w_j x_{7j} \leq 6.5 \\
                  & \sum_{j=1}^{20} w_j x_{8j} \leq 7.5 \\
                  & \sum_{j=1}^{20} w_j x_{9j} \leq 8.2 \\
                  & \sum_{j=1}^{20} w_j x_{10j} \leq 5.7 \\
                  & x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\end{align*}
$$

Where $v_j$ and $w_j$ are as given in the table above, and shelf and product indices correspond to the source order.