Let $x_{ij}$ be the number of units of product $j$ (ProductName from products.csv) to be placed on shelf $i$ (ShelfID from capacity.csv). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $S$ = set of shelves (ShelfID): $\{1,2,3,4,5,6,7,8,9,10\}$
- $P$ = set of products (ProductName): 
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

- $v_j$ = Value of product $j$:

| ProductName             | Value |
|-------------------------|-------|
| Smartphone              | 200   |
| Laptop                  | 1500  |
| Headphones              | 100   |
| Camera                  | 800   |
| Smartwatch              | 250   |
| Tablet                  | 600   |
| Bluetooth Speaker       | 150   |
| Keyboard                | 80    |
| Mouse                   | 50    |
| Monitor                 | 300   |
| Printer                 | 400   |
| External Hard Drive     | 120   |
| Router                  | 60    |
| Power Bank              | 40    |
| Memory Card             | 30    |
| USB Flash Drive         | 25    |
| Smart Home Hub          | 100   |
| Gaming Console          | 500   |
| Fitness Tracker         | 90    |
| E-Reader                | 180   |

- $w_j$ = Weight of product $j$:

| ProductName             | Weight |
|-------------------------|--------|
| Smartphone              | 1.0    |
| Laptop                  | 5.0    |
| Headphones              | 0.5    |
| Camera                  | 2.0    |
| Smartwatch              | 0.3    |
| Tablet                  | 1.5    |
| Bluetooth Speaker       | 1.0    |
| Keyboard                | 0.8    |
| Mouse                   | 0.2    |
| Monitor                 | 3.0    |
| Printer                 | 4.0    |
| External Hard Drive     | 0.5    |
| Router                  | 0.3    |
| Power Bank              | 0.4    |
| Memory Card             | 0.05   |
| USB Flash Drive         | 0.02   |
| Smart Home Hub          | 0.6    |
| Gaming Console          | 4.0    |
| Fitness Tracker         | 0.2    |
| E-Reader                | 0.5    |

- $C_i$ = Capacity of shelf $i$:

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

---

**Mathematical Model:**

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
$$

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

**All identifiers and coefficients are as retrieved and preserved in original order.**