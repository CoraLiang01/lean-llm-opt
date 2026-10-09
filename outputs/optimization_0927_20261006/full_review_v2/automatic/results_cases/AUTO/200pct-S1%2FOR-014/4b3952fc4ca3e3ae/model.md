Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all shelves $i$ and products $j$.

Define:
- $S$ = set of shelves, indexed by ShelfID: $\{1,2,3,4,5,6,7,8,9,10\}$
- $P$ = set of products, indexed by ProductName: 
  - Smartphone
  - Laptop
  - Headphones
  - Camera
  - Smartwatch
  - Tablet
  - Bluetooth Speaker
  - Keyboard
  - Mouse
  - Monitor
  - Printer
  - External Hard Drive
  - Router
  - Power Bank
  - Memory Card
  - USB Flash Drive
  - Smart Home Hub
  - Gaming Console
  - Fitness Tracker
  - E-Reader

Let $v_j$ be the value of product $j$ and $w_j$ be the weight of product $j$ (from products.csv). Let $C_i$ be the capacity of shelf $i$ (from capacity.csv).

##### Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

##### Subject to:

For each shelf $i \in S$:
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
\]

For all $i \in S$, $j \in P$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

#### Data

**Shelves and Capacities (capacity.csv):**
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

**Products, Values, and Weights (products.csv):**
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

**Complete Mathematical Model:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i, \quad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10,\ j = 1,\ldots,20
\end{align*}
\]

Where $v_j$ and $w_j$ are as listed above, and $C_i$ is the capacity for shelf $i$ as listed above.