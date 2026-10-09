Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$. All $x_{ij}$ are nonnegative integers.

Let $S$ be the set of shelves (from capacity.csv, using ShelfID):

$S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$

Let $P$ be the set of products (from products.csv, using ProductName):

$P = \{$
Smartphone,
Laptop,
Headphones,
Camera,
Smartwatch,
Tablet,
Bluetooth Speaker,
Keyboard,
Mouse,
Monitor,
Printer,
External Hard Drive,
Router,
Power Bank,
Memory Card,
USB Flash Drive,
Smart Home Hub,
Gaming Console,
Fitness Tracker,
E-Reader
$\}$

Let $v_j$ be the value of product $j$ (from Value column in products.csv).

Let $w_j$ be the weight of product $j$ (from Weight column in products.csv).

Let $C_i$ be the capacity of shelf $i$ (from Capacity column in capacity.csv).

The first product in products.csv is "Smartphone".

The model is:

Maximize total value:
$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

Subject to:

Shelf capacity constraints (for each shelf $i$):
$$
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in S
$$

Minimum total quantity of the first product ("Smartphone"):
$$
\sum_{i \in S} x_{i,\text{Smartphone}} \geq 5
$$

Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
$$

Where the parameters are:

Shelves (from capacity.csv, in order):

| ShelfID | Capacity |
|---------|----------|
| 1       | 5        |
| 2       | 7        |
| 3       | 6        |
| 4       | 8        |
| 5       | 5.5      |
| 6       | 9        |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

Products (from products.csv, in order):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Smartphone            | 200   | 1      |
| Laptop                | 1500  | 5      |
| Headphones            | 100   | 0.5    |
| Camera                | 800   | 2      |
| Smartwatch            | 250   | 0.3    |
| Tablet                | 600   | 1.5    |
| Bluetooth Speaker     | 150   | 1      |
| Keyboard              | 80    | 0.8    |
| Mouse                 | 50    | 0.2    |
| Monitor               | 300   | 3      |
| Printer               | 400   | 4      |
| External Hard Drive   | 120   | 0.5    |
| Router                | 60    | 0.3    |
| Power Bank            | 40    | 0.4    |
| Memory Card           | 30    | 0.05   |
| USB Flash Drive       | 25    | 0.02   |
| Smart Home Hub        | 100   | 0.6    |
| Gaming Console        | 500   | 4      |
| Fitness Tracker       | 90    | 0.2    |
| E-Reader              | 180   | 0.5    |

Decision variables:
$x_{ij}$: number of units of product $j$ placed on shelf $i$, for all $i \in S$, $j \in P$.

All parameters and indices are as above, using the exact identifiers and values from the retrieved data.