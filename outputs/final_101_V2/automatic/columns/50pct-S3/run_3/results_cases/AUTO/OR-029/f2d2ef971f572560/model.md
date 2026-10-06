Let $x_{ij}$ be the number of units of product $j$ placed on display $i$, where $i$ indexes ShelfID from 1 to 10 and $j$ indexes ProductName in the order given.

Objective:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$
where $v_j$ is the Value of product $j$.

Subject to:

1. Display Capacity Constraints (for each display $i$):
$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i=1,\ldots,10
$$
where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of display $i$.

2. Minimum Quantity of First Product (Smartphone) Across All Displays:
$$
\sum_{i=1}^{10} x_{i,1} \geq 5
$$

3. Nonnegativity and Integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10;\ j=1,\ldots,20
$$

Parameter values (in source order):

Displays (from capacity.csv):

| $i$ | ShelfID | Capacity |
|---|---------|----------|
| 1 | 1       | 5        |
| 2 | 2       | 7        |
| 3 | 3       | 6        |
| 4 | 4       | 8        |
| 5 | 5       | 5.5      |
| 6 | 6       | 9        |
| 7 | 7       | 6.5      |
| 8 | 8       | 7.5      |
| 9 | 9       | 8.2      |
|10 | 10      | 5.7      |

Products (from products.csv):

| $j$ | ProductName             | Value | Weight |
|----|-------------------------|-------|--------|
| 1  | Smartphone              | 200   | 1      |
| 2  | Laptop                  | 1500  | 5      |
| 3  | Headphones              | 100   | 0.5    |
| 4  | Camera                  | 800   | 2      |
| 5  | Smartwatch              | 250   | 0.3    |
| 6  | Tablet                  | 600   | 1.5    |
| 7  | Bluetooth Speaker       | 150   | 1      |
| 8  | Keyboard                | 80    | 0.8    |
| 9  | Mouse                   | 50    | 0.2    |
|10  | Monitor                 | 300   | 3      |
|11  | Printer                 | 400   | 4      |
|12  | External Hard Drive     | 120   | 0.5    |
|13  | Router                  | 60    | 0.3    |
|14  | Power Bank              | 40    | 0.4    |
|15  | Memory Card             | 30    | 0.05   |
|16  | USB Flash Drive         | 25    | 0.02   |
|17  | Smart Home Hub          | 100   | 0.6    |
|18  | Gaming Console          | 500   | 4      |
|19  | Fitness Tracker         | 90    | 0.2    |
|20  | E-Reader                | 180   | 0.5    |

Where:
- $x_{ij}$: number of units of product $j$ placed on display $i$ (nonnegative integer)
- $v_j$: Value of product $j$ (see table)
- $w_j$: Weight of product $j$ (see table)
- $C_i$: Capacity of display $i$ (see table)

All indices, coefficients, and constraints are as retrieved and in original order.