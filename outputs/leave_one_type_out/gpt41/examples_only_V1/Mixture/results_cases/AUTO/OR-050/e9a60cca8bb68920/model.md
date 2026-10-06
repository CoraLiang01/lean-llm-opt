Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i \in \{1,2,\ldots,10\}$ (ShelfID from capacity.csv) and $j \in \{1,2,\ldots,20\}$ (ProductName from products.csv, in source order).

Parameters:
- $c_i$: Capacity of shelf $i$ (from capacity.csv)
- $v_j$: Value of product $j$ (from products.csv)
- $w_j$: Weight of product $j$ (from products.csv)

ShelfID and ProductName in source order:
- Shelves ($i$): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Products ($j$): 
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

Data:
- Shelf capacities ($c_i$): 
  - $c_1 = 5.0$
  - $c_2 = 7.0$
  - $c_3 = 6.0$
  - $c_4 = 8.0$
  - $c_5 = 5.5$
  - $c_6 = 9.0$
  - $c_7 = 6.5$
  - $c_8 = 7.5$
  - $c_9 = 8.2$
  - $c_{10} = 5.7$

- Product values ($v_j$) and weights ($w_j$):

| $j$ | ProductName            | $v_j$ | $w_j$  |
|-----|------------------------|-------|--------|
| 1   | Smartphone             | 200   | 1.0    |
| 2   | Laptop                 | 1500  | 5.0    |
| 3   | Headphones             | 100   | 0.5    |
| 4   | Camera                 | 800   | 2.0    |
| 5   | Smartwatch             | 250   | 0.3    |
| 6   | Tablet                 | 600   | 1.5    |
| 7   | Bluetooth Speaker      | 150   | 1.0    |
| 8   | Keyboard               | 80    | 0.8    |
| 9   | Mouse                  | 50    | 0.2    |
| 10  | Monitor                | 300   | 3.0    |
| 11  | Printer                | 400   | 4.0    |
| 12  | External Hard Drive    | 120   | 0.5    |
| 13  | Router                 | 60    | 0.3    |
| 14  | Power Bank             | 40    | 0.4    |
| 15  | Memory Card            | 30    | 0.05   |
| 16  | USB Flash Drive        | 25    | 0.02   |
| 17  | Smart Home Hub         | 100   | 0.6    |
| 18  | Gaming Console         | 500   | 4.0    |
| 19  | Fitness Tracker        | 90    | 0.2    |
| 20  | E-Reader               | 180   | 0.5    |

Model:

Objective:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

Subject to:

Shelf capacity constraints (for each shelf $i$):
$$
\sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,2,\ldots,10\}
$$

Minimum allocation for the first product (Smartphone):
$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\; j \in \{1,2,\ldots,20\}
$$

Where:
- $x_{ij}$: number of units of product $j$ placed on shelf $i$
- $v_j$, $w_j$ as above
- $c_i$ as above

All coefficients and identifiers are as retrieved and in source order.