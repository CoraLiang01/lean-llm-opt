Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Let $S$ be the set of shelves (displays), indexed by ShelfID as in the table below.

Let $P$ be the set of products, indexed in the order given in the products.csv table below.

Let $v_j$ be the value of product $j$.

Let $w_j$ be the weight of product $j$.

Let $C_i$ be the capacity of shelf $i$.

Let the first product be "Smartphone".

---

#### Sets

- Shelves $S = \{1,2,3,4,5,6,7,8,9,10\}$
- Products $P = \{$
    1: Smartphone,
    2: Laptop,
    3: Headphones,
    4: Camera,
    5: Smartwatch,
    6: Tablet,
    7: Bluetooth Speaker,
    8: Keyboard,
    9: Mouse,
    10: Monitor,
    11: Printer,
    12: External Hard Drive,
    13: Router,
    14: Power Bank,
    15: Memory Card,
    16: USB Flash Drive,
    17: Smart Home Hub,
    18: Gaming Console,
    19: Fitness Tracker,
    20: E-Reader
$\}$

---

#### Parameters

| ShelfID | Capacity |
|---------|----------|
| 1       | 5        |
| 2       | 7        |
| 3       | 6        |
| 4       | 8        |
| 5       | 5.5      |
| 6       | 9        |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

| ProductName             | Value | Weight |
|-------------------------|-------|--------|
| Smartphone              | 200   | 1      |
| Laptop                  | 1500  | 5      |
| Headphones              | 100   | 0.5    |
| Camera                  | 800   | 2      |
| Smartwatch              | 250   | 0.3    |
| Tablet                  | 600   | 1.5    |
| Bluetooth Speaker       | 150   | 1      |
| Keyboard                | 80    | 0.8    |
| Mouse                   | 50    | 0.2    |
| Monitor                 | 300   | 3      |
| Printer                 | 400   | 4      |
| External Hard Drive     | 120   | 0.5    |
| Router                  | 60    | 0.3    |
| Power Bank              | 40    | 0.4    |
| Memory Card             | 30    | 0.05   |
| USB Flash Drive         | 25    | 0.02   |
| Smart Home Hub          | 100   | 0.6    |
| Gaming Console          | 500   | 4      |
| Fitness Tracker         | 90    | 0.2    |
| E-Reader                | 180   | 0.5    |

---

#### Mathematical Model

**Objective:**
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

**Subject to:**

1. **Shelf Capacity Constraints:**
   \[
   \sum_{j \in P} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in S
   \]

2. **Minimum Quantity of First Product ("Smartphone") Across All Shelves:**
   \[
   \sum_{i \in S} x_{i,1} \geq 5
   \]

3. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
   \]

---

**All parameters and indices are as given in the tables above, preserving original order and identifiers.**