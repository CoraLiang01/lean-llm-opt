Let $x_{ij}$ be the number of units of product $j$ (ProductName) to be placed on shelf $i$ (ShelfID). All $x_{ij}$ are integer and $x_{ij} \geq 0$.

Sets:
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (ShelfID from capacity.csv)
- $j \in \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$ (ProductName from products.csv)

Parameters:
- $v_j$ = Value of product $j$ (from Value column)
- $w_j$ = Weight of product $j$ (from Weight column)
- $C_i$ = Capacity of shelf $i$ (from Capacity column)

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j} v_j \cdot x_{ij}
\]

Subject to (for each shelf $i$):
\[
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:

Shelf capacities (from capacity.csv, in source order):
- ShelfID 1: $C_1 = 5.0$
- ShelfID 2: $C_2 = 7.0$
- ShelfID 3: $C_3 = 6.0$
- ShelfID 4: $C_4 = 8.0$
- ShelfID 5: $C_5 = 5.5$
- ShelfID 6: $C_6 = 9.0$
- ShelfID 7: $C_7 = 6.5$
- ShelfID 8: $C_8 = 7.5$
- ShelfID 9: $C_9 = 8.2$
- ShelfID 10: $C_{10} = 5.7$

Product values and weights (from products.csv, in source order):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Smartphone            | 200   | 1.0    |
| Laptop                | 1500  | 5.0    |
| Headphones            | 100   | 0.5    |
| Camera                | 800   | 2.0    |
| Smartwatch            | 250   | 0.3    |
| Tablet                | 600   | 1.5    |
| Bluetooth Speaker     | 150   | 1.0    |
| Keyboard              | 80    | 0.8    |
| Mouse                 | 50    | 0.2    |
| Monitor               | 300   | 3.0    |
| Printer               | 400   | 4.0    |
| External Hard Drive   | 120   | 0.5    |
| Router                | 60    | 0.3    |
| Power Bank            | 40    | 0.4    |
| Memory Card           | 30    | 0.05   |
| USB Flash Drive       | 25    | 0.02   |
| Smart Home Hub        | 100   | 0.6    |
| Gaming Console        | 500   | 4.0    |
| Fitness Tracker       | 90    | 0.2    |
| E-Reader              | 180   | 0.5    |

All $x_{ij}$ are integer and nonnegative.