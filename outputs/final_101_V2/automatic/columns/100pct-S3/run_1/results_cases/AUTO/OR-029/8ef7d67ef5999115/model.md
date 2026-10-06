Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i$ indexes ShelfID from 1 to 10 and $j$ indexes ProductName in the order given.

**Parameters:**

- $S$ = set of shelves (displays), indexed by $i$ (ShelfID: 1, 2, ..., 10)
- $P$ = set of products, indexed by $j$ (ProductName, in order as below)
- $c_i$ = Capacity of shelf $i$ (from Capacity column of capacity.csv)
- $v_j$ = Value of product $j$ (from Value column of products.csv)
- $w_j$ = Weight of product $j$ (from Weight column of products.csv)

**Product order (for $j=1$ to $20$):**
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

**Data:**

| ShelfID ($i$) | $c_i$ (Capacity) |
|---|---|
| 1 | 5 |
| 2 | 7 |
| 3 | 6 |
| 4 | 8 |
| 5 | 5.5 |
| 6 | 9 |
| 7 | 6.5 |
| 8 | 7.5 |
| 9 | 8.2 |
| 10 | 5.7 |

| ProductName ($j$) | $v_j$ (Value) | $w_j$ (Weight) |
|---|---|---|
| Smartphone | 200 | 1 |
| Laptop | 1500 | 5 |
| Headphones | 100 | 0.5 |
| Camera | 800 | 2 |
| Smartwatch | 250 | 0.3 |
| Tablet | 600 | 1.5 |
| Bluetooth Speaker | 150 | 1 |
| Keyboard | 80 | 0.8 |
| Mouse | 50 | 0.2 |
| Monitor | 300 | 3 |
| Printer | 400 | 4 |
| External Hard Drive | 120 | 0.5 |
| Router | 60 | 0.3 |
| Power Bank | 40 | 0.4 |
| Memory Card | 30 | 0.05 |
| USB Flash Drive | 25 | 0.02 |
| Smart Home Hub | 100 | 0.6 |
| Gaming Console | 500 | 4 |
| Fitness Tracker | 90 | 0.2 |
| E-Reader | 180 | 0.5 |

---

**Mathematical Model:**

Maximize total value:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

Subject to:

1. **Shelf capacity constraints (for each shelf $i$):**
$$
\sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i = 1, \ldots, 10
$$

2. **Minimum allocation of first product (Smartphone) across all shelves:**
$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

3. **Nonnegativity and integrality:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1, \ldots, 10;\; j = 1, \ldots, 20
$$

Where:

- $x_{ij}$ = number of units of product $j$ placed on shelf $i$
- $v_j$, $w_j$ as above
- $c_i$ as above

All coefficients and indices are as retrieved and ordered above.