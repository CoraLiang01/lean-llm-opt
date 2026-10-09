Let:
- $I$ = set of shelves, indexed by $i$, with ShelfID as below.
- $J$ = set of products, indexed by $j$, with ProductName as below.
- $x_{ij}$ = number of units of product $j$ placed on shelf $i$ (decision variable, nonnegative integer).

**Parameters:**

From capacity.csv (in source order):

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

From products.csv (in source order):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Smartphone            | 200   | 1      |
| Laptop                | 1500  | 5      |
| Headphones            | 100   | 0.5    |
| Camera                | 800   | 2      |
| Smartwatch            | 250   | 0.3    |
| Tablet                | 600   | 1.5    |
| Bluetooth Speaker     | 150   | 1      |
| Keyboard              | 80    | 0.8    |
| Mouse                 | 50    | 0.2    |
| Monitor               | 300   | 3      |
| Printer               | 400   | 4      |
| External Hard Drive   | 120   | 0.5    |
| Router                | 60    | 0.3    |
| Power Bank            | 40    | 0.4    |
| Memory Card           | 30    | 0.05   |
| USB Flash Drive       | 25    | 0.02   |
| Smart Home Hub        | 100   | 0.6    |
| Gaming Console        | 500   | 4      |
| Fitness Tracker       | 90    | 0.2    |
| E-Reader              | 180   | 0.5    |

Let $c_i$ = Capacity of shelf $i$ (from above table).

Let $v_j$ = Value of product $j$ (from above table).

Let $w_j$ = Weight of product $j$ (from above table).

Let $j^* =$ index of the first product in the above list, i.e., $j^*$ corresponds to "Smartphone".

---

### Mathematical Model

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, \; j \in J
$$

**Objective:**
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

**Subject to:**

1. **Shelf Capacity Constraints:** For each shelf $i$,
   $$
   \sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
   $$

2. **Minimum Quantity of First Product Across All Shelves:**
   $$
   \sum_{i \in I} x_{i, j^*} \geq 5
   $$

3. **Nonnegativity and Integrality:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I, \; j \in J
   $$

---

**Where:**

- $I = \{1,2,3,4,5,6,7,8,9,10\}$ (ShelfID from capacity.csv, in source order)
- $J =$ (ProductName from products.csv, in source order):

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

- $v_j$ and $w_j$ as in the table above, for each $j$.
- $c_i$ as in the table above, for each $i$.

---

**Summary of Constraints:**
- Each shelf's total product weight cannot exceed its capacity.
- At least 5 units of "Smartphone" (the first product) must be placed in total.
- All $x_{ij}$ are nonnegative integers.