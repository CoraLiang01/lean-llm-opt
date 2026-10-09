Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i \in \{1,2,\ldots,10\}$ and $j \in \{1,2,\ldots,18\}$, corresponding to the order of ShelfID and ProductName as given below.

Let $v_j$ be the value of product $j$, $w_j$ its weight, and $C_i$ the capacity of shelf $i$.

Product order (from products.csv):

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

But only the first 18 products are present in the data above (ending with E-Reader).

Shelf order (from capacity.csv):

1. ShelfID 1, Capacity 5
2. ShelfID 2, Capacity 7
3. ShelfID 3, Capacity 6
4. ShelfID 4, Capacity 8
5. ShelfID 5, Capacity 5.5
6. ShelfID 6, Capacity 9
7. ShelfID 7, Capacity 6.5
8. ShelfID 8, Capacity 7.5
9. ShelfID 9, Capacity 8.2
10. ShelfID 10, Capacity 5.7

Parameters:

- $v_j$ (Value): 
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

- $w_j$ (Weight): 
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

- $C_i$ (Capacity): 
  - Shelf 1: 5
  - Shelf 2: 7
  - Shelf 3: 6
  - Shelf 4: 8
  - Shelf 5: 5.5
  - Shelf 6: 9
  - Shelf 7: 6.5
  - Shelf 8: 7.5
  - Shelf 9: 8.2
  - Shelf 10: 5.7

Mathematical Model:

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]

Subject to:

Capacity constraints for each shelf:
\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
\]

Minimum allocation for the first product (Smartphone):
\[
\sum_{i=1}^{10} x_{i1} \geq 5
\]

Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
\]

Where:

- $x_{ij}$: Number of units of product $j$ placed on shelf $i$
- $v_j$: Value of product $j$ (see above)
- $w_j$: Weight of product $j$ (see above)
- $C_i$: Capacity of shelf $i$ (see above)

All coefficients and identifiers are as retrieved and in original order.