Let $x_{ij}$ be the number of units of product $j$ (ProductName) placed on shelf $i$ (ShelfID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves (from capacity.csv, in order):

  | ShelfID |
  |---------|
  | 1       |
  | 2       |
  | 3       |
  | 4       |
  | 5       |
  | 6       |
  | 7       |
  | 8       |
  | 9       |
  | 10      |

  Shelf capacities:

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

- Products (from products.csv, in order):

  | ProductName            | Value | Weight |
  |----------------------- |-------|--------|
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

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{ShelfID 1, ..., 10}\},\ j \in \{\text{all ProductName above}\}
$$

**Objective:**

$$
\max \sum_{i \in \{\text{ShelfID}\}} \sum_{j \in \{\text{ProductName}\}} \text{Value}_j \cdot x_{ij}
$$

**Subject to:**

For each shelf $i$ (ShelfID):

$$
\sum_{j} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \qquad \forall i \in \{\text{ShelfID 1, ..., 10}\}
$$

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Where:**

- $\text{Value}_j$ and $\text{Weight}_j$ are as given in the products table above.
- $\text{Capacity}_i$ is as given in the shelves table above.
- $x_{ij}$ is the number of units of product $j$ placed on shelf $i$.

All data and identifiers are used as retrieved and in original order.