Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i \in \{1,2,\ldots,10\}$ (corresponding to ShelfID from capacity.csv) and $j \in \{1,2,\ldots,20\}$ (corresponding to the order of ProductName in products.csv, with $j=1$ for "Smartphone", $j=2$ for "Laptop", etc.).

Parameters:

- $v_j$: Value of product $j$ (from products.csv)
- $w_j$: Weight of product $j$ (from products.csv)
- $C_i$: Capacity of shelf $i$ (from capacity.csv)

Data (in source order):

From capacity.csv:

| $i$ | ShelfID | $C_i$ |
|---|---------|-------|
| 1 | 1       | 5.0   |
| 2 | 2       | 7.0   |
| 3 | 3       | 6.0   |
| 4 | 4       | 8.0   |
| 5 | 5       | 5.5   |
| 6 | 6       | 9.0   |
| 7 | 7       | 6.5   |
| 8 | 8       | 7.5   |
| 9 | 9       | 8.2   |
|10 | 10      | 5.7   |

From products.csv (in order):

| $j$ | ProductName             | $v_j$ | $w_j$  |
|---|--------------------------|-------|--------|
| 1 | Smartphone               | 200   | 1.0    |
| 2 | Laptop                   | 1500  | 5.0    |
| 3 | Headphones               | 100   | 0.5    |
| 4 | Camera                   | 800   | 2.0    |
| 5 | Smartwatch               | 250   | 0.3    |
| 6 | Tablet                   | 600   | 1.5    |
| 7 | Bluetooth Speaker        | 150   | 1.0    |
| 8 | Keyboard                 | 80    | 0.8    |
| 9 | Mouse                    | 50    | 0.2    |
|10 | Monitor                  | 300   | 3.0    |
|11 | Printer                  | 400   | 4.0    |
|12 | External Hard Drive      | 120   | 0.5    |
|13 | Router                   | 60    | 0.3    |
|14 | Power Bank               | 40    | 0.4    |
|15 | Memory Card              | 30    | 0.05   |
|16 | USB Flash Drive          | 25    | 0.02   |
|17 | Smart Home Hub           | 100   | 0.6    |
|18 | Gaming Console           | 500   | 4.0    |
|19 | Fitness Tracker          | 90    | 0.2    |
|20 | E-Reader                 | 180   | 0.5    |

Model:

Maximize total value:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

Subject to:

1. Shelf capacity constraints (for each shelf $i$):
$$
\sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i = 1,\ldots,10
$$

2. Minimum total quantity of the first product ("Smartphone") across all shelves:
$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

3. Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10,\; j = 1,\ldots,20
$$

Where:

- $v_j$, $w_j$ as above for each product $j$ in the order given.
- $C_i$ as above for each shelf $i$ in the order given.