Let $x_{ij}$ be the number of units of product $j$ (with ProductName as below) to be placed on shelf $i$ (with ShelfID as below). All $x_{ij}$ are nonnegative integers.

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

  Shelf capacities (Capacity column):

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

  | ProductName             | Value | Weight |
  |------------------------ |-------|--------|
  | Smartphone              | 200   | 1.0    |
  | Laptop                  | 1500  | 5.0    |
  | Headphones              | 100   | 0.5    |
  | Camera                  | 800   | 2.0    |
  | Smartwatch              | 250   | 0.3    |
  | Tablet                  | 600   | 1.5    |
  | Bluetooth Speaker       | 150   | 1.0    |
  | Keyboard                | 80    | 0.8    |
  | Mouse                   | 50    | 0.2    |
  | Monitor                 | 300   | 3.0    |
  | Printer                 | 400   | 4.0    |
  | External Hard Drive     | 120   | 0.5    |
  | Router                  | 60    | 0.3    |
  | Power Bank              | 40    | 0.4    |
  | Memory Card             | 30    | 0.05   |
  | USB Flash Drive         | 25    | 0.02   |
  | Smart Home Hub          | 100   | 0.6    |
  | Gaming Console          | 500   | 4.0    |
  | Fitness Tracker         | 90    | 0.2    |
  | E-Reader                | 180   | 0.5    |

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} = \text{number of units of product } j \text{ placed on shelf } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

where $i \in \{1,2,\ldots,10\}$ (ShelfID), $j$ indexes the 20 products in the order above.

---

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$ as listed above.

---

**Constraints:**

For each shelf $i$ (with ShelfID as above):

$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
$$

where $w_j$ is the Weight of product $j$ as listed above, and $C_i$ is the Capacity of shelf $i$ as listed above.

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Explicit Data Table Reference**

- ShelfID and Capacity (in order):

  1: 5.0  2: 7.0  3: 6.0  4: 8.0  5: 5.5  6: 9.0  7: 6.5  8: 7.5  9: 8.2  10: 5.7

- ProductName, Value, Weight (in order):

  1. Smartphone: 200, 1.0  2. Laptop: 1500, 5.0  3. Headphones: 100, 0.5  4. Camera: 800, 2.0  5. Smartwatch: 250, 0.3  6. Tablet: 600, 1.5  7. Bluetooth Speaker: 150, 1.0  8. Keyboard: 80, 0.8  9. Mouse: 50, 0.2  10. Monitor: 300, 3.0  11. Printer: 400, 4.0  12. External Hard Drive: 120, 0.5  13. Router: 60, 0.3  14. Power Bank: 40, 0.4  15. Memory Card: 30, 0.05  16. USB Flash Drive: 25, 0.02  17. Smart Home Hub: 100, 0.6  18. Gaming Console: 500, 4.0  19. Fitness Tracker: 90, 0.2  20. E-Reader: 180, 0.5

---

**Summary:**

Maximize total value of products allocated to shelves, subject to each shelf's total weight not exceeding its capacity, with integer, nonnegative allocation variables for each product-shelf pair. All data and identifiers are as retrieved and ordered above.