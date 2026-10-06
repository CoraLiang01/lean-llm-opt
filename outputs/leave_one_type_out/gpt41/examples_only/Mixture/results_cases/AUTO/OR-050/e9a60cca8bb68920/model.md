Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i$ indexes ShelfID from 1 to 10 and $j$ indexes ProductName in the order given.

Parameters:
- $v_j$: Value of product $j$
- $w_j$: Weight of product $j$
- $C_i$: Capacity of shelf $i$

Data:

Shelves (from capacity.csv):

| $i$ | ShelfID | $C_i$ |
|----|---------|-------|
| 1  | 1       | 5.0   |
| 2  | 2       | 7.0   |
| 3  | 3       | 6.0   |
| 4  | 4       | 8.0   |
| 5  | 5       | 5.5   |
| 6  | 6       | 9.0   |
| 7  | 7       | 6.5   |
| 8  | 8       | 7.5   |
| 9  | 9       | 8.2   |
| 10 | 10      | 5.7   |

Products (from products.csv):

| $j$ | ProductName           | $v_j$ | $w_j$  |
|-----|-----------------------|-------|--------|
| 1   | Smartphone            | 200   | 1.0    |
| 2   | Laptop                | 1500  | 5.0    |
| 3   | Headphones            | 100   | 0.5    |
| 4   | Camera                | 800   | 2.0    |
| 5   | Smartwatch            | 250   | 0.3    |
| 6   | Tablet                | 600   | 1.5    |
| 7   | Bluetooth Speaker     | 150   | 1.0    |
| 8   | Keyboard              | 80    | 0.8    |
| 9   | Mouse                 | 50    | 0.2    |
| 10  | Monitor               | 300   | 3.0    |
| 11  | Printer               | 400   | 4.0    |
| 12  | External Hard Drive   | 120   | 0.5    |
| 13  | Router                | 60    | 0.3    |
| 14  | Power Bank            | 40    | 0.4    |
| 15  | Memory Card           | 30    | 0.05   |
| 16  | USB Flash Drive       | 25    | 0.02   |
| 17  | Smart Home Hub        | 100   | 0.6    |
| 18  | Gaming Console        | 500   | 4.0    |
| 19  | Fitness Tracker       | 90    | 0.2    |
| 20  | E-Reader              | 180   | 0.5    |

Model:

Objective:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

Subject to:

1. Shelf capacity constraints (for each shelf $i$):
$$
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
$$

2. Minimum allocation of the first product (Smartphone) across all shelves:
$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

3. Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
$$

Where:
- $x_{ij}$: number of units of product $j$ placed on shelf $i$
- $v_j$: value of product $j$ (see table above)
- $w_j$: weight of product $j$ (see table above)
- $C_i$: capacity of shelf $i$ (see table above)

All indices and coefficients are as given in the source data, in original order.