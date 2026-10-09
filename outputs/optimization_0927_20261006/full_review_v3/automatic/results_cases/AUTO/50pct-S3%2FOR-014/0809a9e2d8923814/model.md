Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all shelves $i$ and products $j$.

Let $S$ be the set of shelves (indexed by ShelfID), and $P$ be the set of products (indexed by ProductName).

Let $v_j$ be the value of product $j$ (from the "Value" column), and $w_j$ be the weight of product $j$ (from the "Weight" column).

Let $C_i$ be the capacity of shelf $i$ (from the "Capacity" column).

---

**Sets:**

- Shelves $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$
- Products $P = \{$
    Smartphone,
    Laptop,
    Headphones,
    Camera,
    Smartwatch,
    Tablet,
    Bluetooth Speaker,
    Keyboard,
    Mouse,
    Monitor,
    Printer,
    External Hard Drive,
    Router,
    Power Bank,
    Memory Card,
    USB Flash Drive,
    Smart Home Hub,
    Gaming Console,
    Fitness Tracker,
    E-Reader
  $\}$

**Parameters:**

- Shelf capacities $C_i$:

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

- Product values $v_j$ and weights $w_j$:

  | ProductName           | Value | Weight |
  |----------------------|-------|--------|
  | Smartphone           | 200   | 1.0    |
  | Laptop               | 1500  | 5.0    |
  | Headphones           | 100   | 0.5    |
  | Camera               | 800   | 2.0    |
  | Smartwatch           | 250   | 0.3    |
  | Tablet               | 600   | 1.5    |
  | Bluetooth Speaker    | 150   | 1.0    |
  | Keyboard             | 80    | 0.8    |
  | Mouse                | 50    | 0.2    |
  | Monitor              | 300   | 3.0    |
  | Printer              | 400   | 4.0    |
  | External Hard Drive  | 120   | 0.5    |
  | Router               | 60    | 0.3    |
  | Power Bank           | 40    | 0.4    |
  | Memory Card          | 30    | 0.05   |
  | USB Flash Drive      | 25    | 0.02   |
  | Smart Home Hub       | 100   | 0.6    |
  | Gaming Console       | 500   | 4.0    |
  | Fitness Tracker      | 90    | 0.2    |
  | E-Reader             | 180   | 0.5    |

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

**Where:**

- $x_{ij}$: Number of units of product $j$ placed on shelf $i$ (integer, $\geq 0$)
- $v_j$: Value of product $j$ (see table above)
- $w_j$: Weight of product $j$ (see table above)
- $C_i$: Capacity of shelf $i$ (see table above)

All data and identifiers are as retrieved and preserved in original order.