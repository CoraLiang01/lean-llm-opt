Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i$ indexes ShelfID from the capacity.csv file (in source order), and $j$ indexes ProductName from the products.csv file (in source order). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $S$ be the set of shelves (displays), with ShelfID as below (in source order):
  1. 1
  2. 2
  3. 3
  4. 4
  5. 5
  6. 6
  7. 7
  8. 8
  9. 9
  10. 10

- Let $P$ be the set of products, with ProductName as below (in source order):
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
  - Smartphone: 1
  - Laptop: 5
  - Headphones: 0.5
  - Camera: 2
  - Smartwatch: 0.3
  - Tablet: 1.5
  - Bluetooth Speaker: 1
  - Keyboard: 0.8
  - Mouse: 0.2
  - Monitor: 3
  - Printer: 4
  - External Hard Drive: 0.5
  - Router: 0.3
  - Power Bank: 0.4
  - Memory Card: 0.05
  - USB Flash Drive: 0.02
  - Smart Home Hub: 0.6
  - Gaming Console: 4
  - Fitness Tracker: 0.2
  - E-Reader: 0.5

- Let $C_i$ be the capacity of shelf $i$:
  - ShelfID 1: 5
  - ShelfID 2: 7
  - ShelfID 3: 6
  - ShelfID 4: 8
  - ShelfID 5: 5.5
  - ShelfID 6: 9
  - ShelfID 7: 6.5
  - ShelfID 8: 7.5
  - ShelfID 9: 8.2
  - ShelfID 10: 5.7

**Model:**

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j x_{ij}
\]

Subject to:

1. Shelf capacity constraints (for each shelf $i$):
\[
\sum_{j \in P} w_j x_{ij} \leq C_i \qquad \forall i \in S
\]

2. Minimum total quantity of the first product (Smartphone) across all shelves:
\[
\sum_{i \in S} x_{i,1} \geq 5
\]

3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
\]

**Where:**

- $x_{ij}$: Number of units of product $j$ placed on shelf $i$ (nonnegative integer)
- $v_j$: Value of product $j$ (see above)
- $w_j$: Weight of product $j$ (see above)
- $C_i$: Capacity of shelf $i$ (see above)
- $S$: Set of ShelfIDs (1, 2, ..., 10)
- $P$: Set of ProductNames (in the order listed above; $j=1$ is Smartphone)

All identifiers and coefficients are as retrieved and in original order.