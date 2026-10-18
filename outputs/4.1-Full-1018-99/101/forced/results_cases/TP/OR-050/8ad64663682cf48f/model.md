Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of displays (ShelfID), and $J = \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$ be the set of products, in the order given.

Let $x_{ij} \geq 0$ (integer) be the number of units of product $j$ placed on display $i$.

Parameters (from retrieved data):

Display capacities:
- Shelf 1: $C_1 = 5.0$
- Shelf 2: $C_2 = 7.0$
- Shelf 3: $C_3 = 6.0$
- Shelf 4: $C_4 = 8.0$
- Shelf 5: $C_5 = 5.5$
- Shelf 6: $C_6 = 9.0$
- Shelf 7: $C_7 = 6.5$
- Shelf 8: $C_8 = 7.5$
- Shelf 9: $C_9 = 8.2$
- Shelf 10: $C_{10} = 5.7$

Product values and weights (in source order):

| $j$ | Product Name           | $v_j$ (Value) | $w_j$ (Weight) |
|-----|------------------------|---------------|----------------|
| 1   | Smartphone             | 200           | 1.0            |
| 2   | Laptop                 | 1500          | 5.0            |
| 3   | Headphones             | 100           | 0.5            |
| 4   | Camera                 | 800           | 2.0            |
| 5   | Smartwatch             | 250           | 0.3            |
| 6   | Tablet                 | 600           | 1.5            |
| 7   | Bluetooth Speaker      | 150           | 1.0            |
| 8   | Keyboard               | 80            | 0.8            |
| 9   | Mouse                  | 50            | 0.2            |
| 10  | Monitor                | 300           | 3.0            |
| 11  | Printer                | 400           | 4.0            |
| 12  | External Hard Drive    | 120           | 0.5            |
| 13  | Router                 | 60            | 0.3            |
| 14  | Power Bank             | 40            | 0.4            |
| 15  | Memory Card            | 30            | 0.05           |
| 16  | USB Flash Drive        | 25            | 0.02           |
| 17  | Smart Home Hub         | 100           | 0.6            |
| 18  | Gaming Console         | 500           | 4.0            |
| 19  | Fitness Tracker        | 90            | 0.2            |
| 20  | E-Reader               | 180           | 0.5            |

Model:

Maximize total value:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

Subject to:

1. Capacity constraints for each display $i$:
$$
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i=1,\ldots,10
$$

2. Minimum total quantity of the first product (Smartphone) across all displays:
$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

3. Nonnegativity and integrality:
$$
x_{ij} \geq 0,\quad x_{ij} \in \mathbb{Z} \qquad \forall i=1,\ldots,10;\ j=1,\ldots,20
$$

All parameters ($C_i$, $v_j$, $w_j$) are as listed above, with indices matching the source order.

Retrieved Information:

Display capacities (capacity.csv, source order):

- ShelfID 1: Capacity 5.0
- ShelfID 2: Capacity 7.0
- ShelfID 3: Capacity 6.0
- ShelfID 4: Capacity 8.0
- ShelfID 5: Capacity 5.5
- ShelfID 6: Capacity 9.0
- ShelfID 7: Capacity 6.5
- ShelfID 8: Capacity 7.5
- ShelfID 9: Capacity 8.2
- ShelfID 10: Capacity 5.7

Product values and weights (products.csv, source order):

1. Smartphone: Value 200, Weight 1.0
2. Laptop: Value 1500, Weight 5.0
3. Headphones: Value 100, Weight 0.5
4. Camera: Value 800, Weight 2.0
5. Smartwatch: Value 250, Weight 0.3
6. Tablet: Value 600, Weight 1.5
7. Bluetooth Speaker: Value 150, Weight 1.0
8. Keyboard: Value 80, Weight 0.8
9. Mouse: Value 50, Weight 0.2
10. Monitor: Value 300, Weight 3.0
11. Printer: Value 400, Weight 4.0
12. External Hard Drive: Value 120, Weight 0.5
13. Router: Value 60, Weight 0.3
14. Power Bank: Value 40, Weight 0.4
15. Memory Card: Value 30, Weight 0.05
16. USB Flash Drive: Value 25, Weight 0.02
17. Smart Home Hub: Value 100, Weight 0.6
18. Gaming Console: Value 500, Weight 4.0
19. Fitness Tracker: Value 90, Weight 0.2
20. E-Reader: Value 180, Weight 0.5