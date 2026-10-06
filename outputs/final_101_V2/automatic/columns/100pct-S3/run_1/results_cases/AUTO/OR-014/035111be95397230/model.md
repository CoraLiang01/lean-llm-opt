Let $x_{ij}$ be the number of units of product $j$ (ProductName) to be placed on shelf $i$ (ShelfID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves ($i$):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (from ShelfID in capacity.csv)

- Products ($j$):  
  Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader (from ProductName in products.csv)

- $v_j$: Value of product $j$ (from Value in products.csv)
- $w_j$: Weight of product $j$ (from Weight in products.csv)
- $C_i$: Capacity of shelf $i$ (from Capacity in capacity.csv)

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j \in \text{Products}} v_j \cdot x_{ij}
$$

**Subject to:**

For each shelf $i$ (ShelfID):

$$
\sum_{j \in \text{Products}} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
$$

For all shelves $i$ and products $j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Parameter Data (as retrieved):**

capacity.csv

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

products.csv

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

**Decision variables:**  
$x_{ij}$: number of units of product $j$ to place on shelf $i$, for all $i$ and $j$, $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j \in \text{Products}} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j \in \text{Products}} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, \forall j
\end{align*}
$$

Where all parameters and indices are as listed above, using the exact identifiers and coefficients from the retrieved data.