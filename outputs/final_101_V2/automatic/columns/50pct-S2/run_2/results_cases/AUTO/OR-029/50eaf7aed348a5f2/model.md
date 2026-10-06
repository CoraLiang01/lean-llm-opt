Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where:

- $i$ indexes ShelfID from capacity.csv: $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- $j$ indexes ProductName from products.csv, in the given order:
  1. Smartphone
  2. Laptop
  3. Headphones
  4. Camera
  5. Smartwatch
  6. Tablet
  7. Bluetooth Speaker
  8. Keyboard
  9. Mouse
  10. Monitor
  11. Printer
  12. External Hard Drive
  13. Router
  14. Power Bank
  15. Memory Card
  16. USB Flash Drive
  17. Smart Home Hub
  18. Gaming Console
  19. Fitness Tracker
  20. E-Reader

Parameters:

- $v_j$ = Value of product $j$ (see table below)
- $w_j$ = Weight of product $j$ (see table below)
- $C_i$ = Capacity of shelf $i$ (see table below)

#### Objective:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

#### Subject to:

**1. Shelf capacity constraints (for each shelf $i$):**
$$
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

**2. Minimum allocation of first product (Smartphone) across all shelves:**
$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

**3. Nonnegativity and integrality:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

---

#### Data

**Shelf Capacities (from capacity.csv):**

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

**Product Values and Weights (from products.csv):**

| $j$ | ProductName            | $v_j$ (Value) | $w_j$ (Weight) |
|-----|------------------------|---------------|----------------|
| 1   | Smartphone             | 200           | 1              |
| 2   | Laptop                 | 1500          | 5              |
| 3   | Headphones             | 100           | 0.5            |
| 4   | Camera                 | 800           | 2              |
| 5   | Smartwatch             | 250           | 0.3            |
| 6   | Tablet                 | 600           | 1.5            |
| 7   | Bluetooth Speaker      | 150           | 1              |
| 8   | Keyboard               | 80            | 0.8            |
| 9   | Mouse                  | 50            | 0.2            |
| 10  | Monitor                | 300           | 3              |
| 11  | Printer                | 400           | 4              |
| 12  | External Hard Drive    | 120           | 0.5            |
| 13  | Router                 | 60            | 0.3            |
| 14  | Power Bank             | 40            | 0.4            |
| 15  | Memory Card            | 30            | 0.05           |
| 16  | USB Flash Drive        | 25            | 0.02           |
| 17  | Smart Home Hub         | 100           | 0.6            |
| 18  | Gaming Console         | 500           | 4              |
| 19  | Fitness Tracker        | 90            | 0.2            |
| 20  | E-Reader               | 180           | 0.5            |