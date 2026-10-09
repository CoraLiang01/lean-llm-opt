Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i \in \{\text{1}, \text{2}, \ldots, \text{10}\}$ (ShelfID from capacity.csv, in order), and $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$ (ProductName from products.csv, in order).

Parameters:

- $v_j$ = Value of product $j$ (from Value column)
- $w_j$ = Weight of product $j$ (from Weight column)
- $C_i$ = Capacity of shelf $i$ (from Capacity column)

Numerical values:

Shelves (from capacity.csv, in order):

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

Products (from products.csv, in order):

| $j$ | ProductName             | $v_j$ | $w_j$ |
|-----|-------------------------|-------|-------|
| 1   | Smartphone              | 200   | 1     |
| 2   | Laptop                  | 1500  | 5     |
| 3   | Headphones              | 100   | 0.5   |
| 4   | Camera                  | 800   | 2     |
| 5   | Smartwatch              | 250   | 0.3   |
| 6   | Tablet                  | 600   | 1.5   |
| 7   | Bluetooth Speaker       | 150   | 1     |
| 8   | Keyboard                | 80    | 0.8   |
| 9   | Mouse                   | 50    | 0.2   |
| 10  | Monitor                 | 300   | 3     |
| 11  | Printer                 | 400   | 4     |
| 12  | External Hard Drive     | 120   | 0.5   |
| 13  | Router                  | 60    | 0.3   |
| 14  | Power Bank              | 40    | 0.4   |
| 15  | Memory Card             | 30    | 0.05  |
| 16  | USB Flash Drive         | 25    | 0.02  |
| 17  | Smart Home Hub          | 100   | 0.6   |
| 18  | Gaming Console          | 500   | 4     |
| 19  | Fitness Tracker         | 90    | 0.2   |
| 20  | E-Reader                | 180   | 0.5   |

Model:

Objective:
\[
\max \sum_{i \in \{\text{1},\ldots,\text{10}\}} \sum_{j \in \{\text{Smartphone}, \ldots, \text{E-Reader}\}} v_j x_{ij}
\]

Subject to:

For each shelf $i$ (ShelfID as above):
\[
\sum_{j} w_j x_{ij} \leq C_i \qquad \forall i \in \{\text{1},\ldots,\text{10}\}
\]

Total quantity of the first product (Smartphone) across all shelves:
\[
\sum_{i} x_{i, \text{Smartphone}} \geq 5
\]

Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:

- $x_{ij}$: number of units of product $j$ placed on shelf $i$
- $v_j$: value of product $j$ (see table above)
- $w_j$: weight of product $j$ (see table above)
- $C_i$: capacity of shelf $i$ (see table above)
- $i$ indexes ShelfID in the order: 1, 2, ..., 10
- $j$ indexes ProductName in the order listed above

All coefficients and identifiers are as retrieved and in original order.