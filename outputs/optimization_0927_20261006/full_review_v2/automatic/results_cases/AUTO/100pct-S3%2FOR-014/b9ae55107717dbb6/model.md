Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all shelves $i$ and products $j$.

**Parameters:**

- Shelves (from capacity.csv, in order):

  | ShelfID |
  |---------|
  |   1     |
  |   2     |
  |   3     |
  |   4     |
  |   5     |
  |   6     |
  |   7     |
  |   8     |
  |   9     |
  |  10     |

  Shelf capacities:

  $C_1 = 5.0$, $C_2 = 7.0$, $C_3 = 6.0$, $C_4 = 8.0$, $C_5 = 5.5$, $C_6 = 9.0$, $C_7 = 6.5$, $C_8 = 7.5$, $C_9 = 8.2$, $C_{10} = 5.7$

- Products (from products.csv, in order):

  | ProductName             | Value | Weight |
  |------------------------|-------|--------|
  | Smartphone             |  200  |  1.0   |
  | Laptop                 | 1500  |  5.0   |
  | Headphones             |  100  |  0.5   |
  | Camera                 |  800  |  2.0   |
  | Smartwatch             |  250  |  0.3   |
  | Tablet                 |  600  |  1.5   |
  | Bluetooth Speaker      |  150  |  1.0   |
  | Keyboard               |   80  |  0.8   |
  | Mouse                  |   50  |  0.2   |
  | Monitor                |  300  |  3.0   |
  | Printer                |  400  |  4.0   |
  | External Hard Drive    |  120  |  0.5   |
  | Router                 |   60  |  0.3   |
  | Power Bank             |   40  |  0.4   |
  | Memory Card            |   30  | 0.05   |
  | USB Flash Drive        |   25  | 0.02   |
  | Smart Home Hub         |  100  |  0.6   |
  | Gaming Console         |  500  |  4.0   |
  | Fitness Tracker        |   90  |  0.2   |
  | E-Reader               |  180  |  0.5   |

Let $P$ be the set of products (as listed above), and $S$ be the set of shelves (ShelfID 1 to 10).

Let $v_j$ be the value of product $j$, and $w_j$ be the weight of product $j$.

Let $C_i$ be the capacity of shelf $i$.

---

### Mathematical Model

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

**Explicit Data Table Reference:**

- Shelves $S$ (ShelfID): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Shelf capacities $C_i$: 5.0, 7.0, 6.0, 8.0, 5.5, 9.0, 6.5, 7.5, 8.2, 5.7
- Products $P$ (ProductName): Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader
- Product values $v_j$: 200, 1500, 100, 800, 250, 600, 150, 80, 50, 300, 400, 120, 60, 40, 30, 25, 100, 500, 90, 180
- Product weights $w_j$: 1.0, 5.0, 0.5, 2.0, 0.3, 1.5, 1.0, 0.8, 0.2, 3.0, 4.0, 0.5, 0.3, 0.4, 0.05, 0.02, 0.6, 4.0, 0.2, 0.5

---

**Decision variables:**

$x_{ij}$ = number of units of product $j$ to be placed on shelf $i$, for all $i \in S$, $j \in P$, with $x_{ij} \in \mathbb{Z}_{\geq 0}$.