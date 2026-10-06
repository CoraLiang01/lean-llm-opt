Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

Indices:
- $i \in \{1,2,\ldots,10\}$ (ShelfID from capacity.csv)
- $j \in \{1,2,\ldots,20\}$ (Product order from products.csv, in source order)

Parameters:
- $c_i$ = Capacity of shelf $i$ (from capacity.csv, column "Capacity")
- $v_j$ = Value of product $j$ (from products.csv, column "Value")
- $w_j$ = Weight of product $j$ (from products.csv, column "Weight")

Product order (j=1 is "Smartphone", j=2 is "Laptop", etc.):
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

Shelf order (i=1 is ShelfID 1, etc.):
1. 5
2. 7
3. 6
4. 8
5. 5.5
6. 9
7. 6.5
8. 7.5
9. 8.2
10. 5.7

Parameters (in order):

| $i$ (ShelfID) | $c_i$ (Capacity) |
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

| $j$ (Product) | $v_j$ (Value) | $w_j$ (Weight) |
|---|---|---|
| 1 | 200 | 1 |
| 2 | 1500 | 5 |
| 3 | 100 | 0.5 |
| 4 | 800 | 2 |
| 5 | 250 | 0.3 |
| 6 | 600 | 1.5 |
| 7 | 150 | 1 |
| 8 | 80 | 0.8 |
| 9 | 50 | 0.2 |
| 10 | 300 | 3 |
| 11 | 400 | 4 |
| 12 | 120 | 0.5 |
| 13 | 60 | 0.3 |
| 14 | 40 | 0.4 |
| 15 | 30 | 0.05 |
| 16 | 25 | 0.02 |
| 17 | 100 | 0.6 |
| 18 | 500 | 4 |
| 19 | 90 | 0.2 |
| 20 | 180 | 0.5 |

Mathematical Model:

Objective:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

Subject to:

Shelf capacity constraints (for each shelf $i$):
$$
\sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10
$$

Minimum allocation of first product ("Smartphone") across all shelves:
$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10;\ j=1,\ldots,20
$$

Where all coefficients and identifiers are as above, in the original source order.