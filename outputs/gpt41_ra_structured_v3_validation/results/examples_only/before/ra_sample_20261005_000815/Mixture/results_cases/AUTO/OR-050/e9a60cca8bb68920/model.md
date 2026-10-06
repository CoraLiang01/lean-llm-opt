Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i$ indexes ShelfID from 1 to 10 and $j$ indexes ProductName in the order given.

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of product $j$.

Subject to:

1. Capacity constraints for each shelf $i$:
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,\ldots,10\}
\]
where $w_j$ is the Weight of product $j$, and $c_i$ is the Capacity of shelf $i$.

2. Minimum allocation of the first product ("Smartphone") across all shelves:
\[
\sum_{i=1}^{10} x_{i,1} \geq 5
\]

3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

Parameters (in source order):

Shelves (from capacity.csv):

| ShelfID | Capacity |
|---------|----------|
| 1       | 5.0      |
| 2       | 7.0      |
| 3       | 6.0      |
| 4       | 8.0      |
| 5       | 5.5      |
| 6       | 9.0      |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

Products (from products.csv):

| $j$ | ProductName           | Value | Weight |
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

Decision variables:
\[
x_{ij} = \text{number of units of product } j \text{ placed on shelf } i,\quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]

All indices, coefficients, and constraints are as retrieved and in source order.