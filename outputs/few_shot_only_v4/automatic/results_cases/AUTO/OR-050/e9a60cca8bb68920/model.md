Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i$ indexes ShelfID (from capacity.csv) and $j$ indexes ProductName (from products.csv). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $S$ be the set of shelves (ShelfID): $S = \{1,2,3,4,5,6,7,8,9,10\}$
- Let $P$ be the set of products (ProductName), in the order given:

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

- Let $v_j$ be the value of product $j$:

  - Smartphone: 200
  - Laptop: 1500
  - Headphones: 100
  - Camera: 800
  - Smartwatch: 250
  - Tablet: 600
  - Bluetooth Speaker: 150
  - Keyboard: 80
  - Mouse: 50
  - Monitor: 300
  - Printer: 400
  - External Hard Drive: 120
  - Router: 60
  - Power Bank: 40
  - Memory Card: 30
  - USB Flash Drive: 25
  - Smart Home Hub: 100
  - Gaming Console: 500
  - Fitness Tracker: 90
  - E-Reader: 180

- Let $w_j$ be the weight of product $j$:

  - Smartphone: 1.0
  - Laptop: 5.0
  - Headphones: 0.5
  - Camera: 2.0
  - Smartwatch: 0.3
  - Tablet: 1.5
  - Bluetooth Speaker: 1.0
  - Keyboard: 0.8
  - Mouse: 0.2
  - Monitor: 3.0
  - Printer: 4.0
  - External Hard Drive: 0.5
  - Router: 0.3
  - Power Bank: 0.4
  - Memory Card: 0.05
  - USB Flash Drive: 0.02
  - Smart Home Hub: 0.6
  - Gaming Console: 4.0
  - Fitness Tracker: 0.2
  - E-Reader: 0.5

- Let $C_i$ be the capacity of shelf $i$:

  - Shelf 1: 5.0
  - Shelf 2: 7.0
  - Shelf 3: 6.0
  - Shelf 4: 8.0
  - Shelf 5: 5.5
  - Shelf 6: 9.0
  - Shelf 7: 6.5
  - Shelf 8: 7.5
  - Shelf 9: 8.2
  - Shelf 10: 5.7

**Model:**

Maximize total value:
$$
\max \sum_{i \in S} \sum_{j \in P} v_j x_{ij}
$$

Subject to:

1. Shelf capacity constraints (for each shelf $i$):
$$
\sum_{j \in P} w_j x_{ij} \leq C_i \qquad \forall i \in S
$$

2. Minimum allocation of the first product (Smartphone) across all shelves:
$$
\sum_{i \in S} x_{i,1} \geq 5
$$

3. Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
$$

Where:

- $x_{ij}$: number of units of product $j$ placed on shelf $i$
- $v_j$: value of product $j$ (see above)
- $w_j$: weight of product $j$ (see above)
- $C_i$: capacity of shelf $i$ (see above)
- $S$: set of ShelfID $\{1,2,3,4,5,6,7,8,9,10\}$
- $P$: set of ProductName (see above, 20 products, $j=1$ is Smartphone)

All coefficients and identifiers are as retrieved and in original order.