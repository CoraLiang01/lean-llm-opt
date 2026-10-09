Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$. All $x_{ij}$ are nonnegative integers.

**Sets and Indices:**

- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (ShelfID)
- $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$ (ProductName)

**Parameters:**

- $v_j$ = Value of product $j$:

  | ProductName             | Value |
  |------------------------|-------|
  | Smartphone             | 200   |
  | Laptop                 | 1500  |
  | Headphones             | 100   |
  | Camera                 | 800   |
  | Smartwatch             | 250   |
  | Tablet                 | 600   |
  | Bluetooth Speaker      | 150   |
  | Keyboard               | 80    |
  | Mouse                  | 50    |
  | Monitor                | 300   |
  | Printer                | 400   |
  | External Hard Drive    | 120   |
  | Router                 | 60    |
  | Power Bank             | 40    |
  | Memory Card            | 30    |
  | USB Flash Drive        | 25    |
  | Smart Home Hub         | 100   |
  | Gaming Console         | 500   |
  | Fitness Tracker        | 90    |
  | E-Reader               | 180   |

- $w_j$ = Weight of product $j$:

  | ProductName             | Weight |
  |------------------------|--------|
  | Smartphone             | 1      |
  | Laptop                 | 5      |
  | Headphones             | 0.5    |
  | Camera                 | 2      |
  | Smartwatch             | 0.3    |
  | Tablet                 | 1.5    |
  | Bluetooth Speaker      | 1      |
  | Keyboard               | 0.8    |
  | Mouse                  | 0.2    |
  | Monitor                | 3      |
  | Printer                | 4      |
  | External Hard Drive    | 0.5    |
  | Router                 | 0.3    |
  | Power Bank             | 0.4    |
  | Memory Card            | 0.05   |
  | USB Flash Drive        | 0.02   |
  | Smart Home Hub         | 0.6    |
  | Gaming Console         | 4      |
  | Fitness Tracker        | 0.2    |
  | E-Reader               | 0.5    |

- $C_i$ = Capacity of shelf $i$:

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

**Decision Variables:**

- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all shelves $i$ and products $j$

---

### Objective Function

\[
\max \sum_{i=1}^{10} \sum_{j \in \text{Products}} v_j x_{ij}
\]

---

### Constraints

1. **Shelf Capacity Constraints** (for each shelf $i$):

   \[
   \sum_{j \in \text{Products}} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
   \]

2. **Minimum Total Quantity for First Product ("Smartphone")**:

   \[
   \sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5
   \]

3. **Nonnegativity and Integrality**:

   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
   \]

---

**Parameter Tables (as retrieved):**

- **Capacity Table (capacity.csv, source order):**

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

- **Products Table (products.csv, source order):**

  | ProductName           | Value | Weight |
  |----------------------|-------|--------|
  | Smartphone           | 200   | 1      |
  | Laptop               | 1500  | 5      |
  | Headphones           | 100   | 0.5    |
  | Camera               | 800   | 2      |
  | Smartwatch           | 250   | 0.3    |
  | Tablet               | 600   | 1.5    |
  | Bluetooth Speaker    | 150   | 1      |
  | Keyboard             | 80    | 0.8    |
  | Mouse                | 50    | 0.2    |
  | Monitor              | 300   | 3      |
  | Printer              | 400   | 4      |
  | External Hard Drive  | 120   | 0.5    |
  | Router               | 60    | 0.3    |
  | Power Bank           | 40    | 0.4    |
  | Memory Card          | 30    | 0.05   |
  | USB Flash Drive      | 25    | 0.02   |
  | Smart Home Hub       | 100   | 0.6    |
  | Gaming Console       | 500   | 4      |
  | Fitness Tracker      | 90    | 0.2    |
  | E-Reader             | 180   | 0.5    |