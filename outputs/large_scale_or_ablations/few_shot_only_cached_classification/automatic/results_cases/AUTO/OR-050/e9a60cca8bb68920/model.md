Let $x_{ij}$ = number of units of product $j$ placed on display (shelf) $i$.

Indices:
- $i \in \{1,2,\ldots,10\}$ (ShelfID from capacity.csv)
- $j \in \{1,2,\ldots,20\}$ (ProductName from products.csv, in source order)

Parameters:
- $c_i$ = Capacity of shelf $i$ (from capacity.csv)
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight of product $j$ (from products.csv)

Product order (for $j$):
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

Shelf order (for $i$):
1. 1 (Capacity 5.0)
2. 2 (Capacity 7.0)
3. 3 (Capacity 6.0)
4. 4 (Capacity 8.0)
5. 5 (Capacity 5.5)
6. 6 (Capacity 9.0)
7. 7 (Capacity 6.5)
8. 8 (Capacity 7.5)
9. 9 (Capacity 8.2)
10. 10 (Capacity 5.7)

Parameters (by product, in order):

| $j$ | ProductName           | $v_j$ | $w_j$  |
|-----|-----------------------|-------|--------|
| 1   | Smartphone           | 200   | 1.0    |
| 2   | Laptop               | 1500  | 5.0    |
| 3   | Headphones           | 100   | 0.5    |
| 4   | Camera               | 800   | 2.0    |
| 5   | Smartwatch           | 250   | 0.3    |
| 6   | Tablet               | 600   | 1.5    |
| 7   | Bluetooth Speaker    | 150   | 1.0    |
| 8   | Keyboard             | 80    | 0.8    |
| 9   | Mouse                | 50    | 0.2    |
| 10  | Monitor              | 300   | 3.0    |
| 11  | Printer              | 400   | 4.0    |
| 12  | External Hard Drive  | 120   | 0.5    |
| 13  | Router               | 60    | 0.3    |
| 14  | Power Bank           | 40    | 0.4    |
| 15  | Memory Card          | 30    | 0.05   |
| 16  | USB Flash Drive      | 25    | 0.02   |
| 17  | Smart Home Hub       | 100   | 0.6    |
| 18  | Gaming Console       | 500   | 4.0    |
| 19  | Fitness Tracker      | 90    | 0.2    |
| 20  | E-Reader             | 180   | 0.5    |

Parameters (by shelf, in order):

| $i$ | ShelfID | $c_i$ |
|-----|---------|-------|
| 1   | 1       | 5.0   |
| 2   | 2       | 7.0   |
| 3   | 3       | 6.0   |
| 4   | 4       | 8.0   |
| 5   | 5       | 5.5   |
| 6   | 6       | 9.0   |
| 7   | 7       | 6.5   |
| 8   | 8       | 7.5   |
| 9   | 9       | 8.2   |
| 10  | 10      | 5.7   |

Model:

Maximize total value:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

Subject to:

Shelf capacity constraints (for each shelf $i$):
$$
\sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10
$$

Minimum allocation of first product (Smartphone) across all shelves:
$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
$$

Where:
- $x_{ij}$ = number of units of product $j$ placed on shelf $i$
- $v_j$, $w_j$ as above
- $c_i$ as above

All coefficients and identifiers are as retrieved and in source order.