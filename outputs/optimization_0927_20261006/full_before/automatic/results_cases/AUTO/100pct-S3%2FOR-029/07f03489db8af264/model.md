Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i$ indexes ShelfID from capacity.csv (in order: 1, 2, ..., 10), and $j$ indexes ProductName from products.csv (in order: Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader).

Parameters:

- $v_j$: Value of product $j$ (from Value in products.csv)
- $w_j$: Weight of product $j$ (from Weight in products.csv)
- $C_i$: Capacity of display $i$ (from Capacity in capacity.csv)

Data (in source order):

Displays (capacity.csv):

| $i$ | ShelfID | $C_i$ |
|----|---------|-------|
| 1  | 1       | 5     |
| 2  | 2       | 7     |
| 3  | 3       | 6     |
| 4  | 4       | 8     |
| 5  | 5       | 5.5   |
| 6  | 6       | 9     |
| 7  | 7       | 6.5   |
| 8  | 8       | 7.5   |
| 9  | 9       | 8.2   |
| 10 | 10      | 5.7   |

Products (products.csv):

| $j$ | ProductName             | $v_j$ | $w_j$ |
|-----|------------------------|-------|-------|
| 1   | Smartphone             | 200   | 1     |
| 2   | Laptop                 | 1500  | 5     |
| 3   | Headphones             | 100   | 0.5   |
| 4   | Camera                 | 800   | 2     |
| 5   | Smartwatch             | 250   | 0.3   |
| 6   | Tablet                 | 600   | 1.5   |
| 7   | Bluetooth Speaker      | 150   | 1     |
| 8   | Keyboard               | 80    | 0.8   |
| 9   | Mouse                  | 50    | 0.2   |
| 10  | Monitor                | 300   | 3     |
| 11  | Printer                | 400   | 4     |
| 12  | External Hard Drive    | 120   | 0.5   |
| 13  | Router                 | 60    | 0.3   |
| 14  | Power Bank             | 40    | 0.4   |
| 15  | Memory Card            | 30    | 0.05  |
| 16  | USB Flash Drive        | 25    | 0.02  |
| 17  | Smart Home Hub         | 100   | 0.6   |
| 18  | Gaming Console         | 500   | 4     |
| 19  | Fitness Tracker        | 90    | 0.2   |
| 20  | E-Reader               | 180   | 0.5   |

Mathematical Model:

Maximize total value:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

Subject to:

Display capacity constraints (for each display $i$):
$$
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1, \ldots, 10
$$

Minimum allocation of the first product (Smartphone) across all displays:
$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1, \ldots, 10;\; j = 1, \ldots, 20
$$

Where:

- $v_j$ and $w_j$ are as given in the table above for each product $j$.
- $C_i$ is as given in the table above for each display $i$.
- $x_{ij}$ is the number of units of product $j$ placed on display $i$.