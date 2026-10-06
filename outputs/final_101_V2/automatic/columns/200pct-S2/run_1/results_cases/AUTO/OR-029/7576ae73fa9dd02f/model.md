Let:
- $i$ index the displays (ShelfID from capacity.csv: $i \in \{1,2,3,4,5,6,7,8,9,10\}$)
- $j$ index the products (ProductName from products.csv, in source order: $j \in \{1,2,\ldots,20\}$, with $j=1$ corresponding to "Smartphone", $j=2$ to "Laptop", etc.)
- $x_{ij}$ = number of units of product $j$ placed on display $i$ (decision variable, nonnegative integer)

Parameters:
- $v_j$ = Value of product $j$ (from products.csv, column "Value")
- $w_j$ = Weight of product $j$ (from products.csv, column "Weight")
- $C_i$ = Capacity of display $i$ (from capacity.csv, column "Capacity")

Data (in source order):

Displays (capacity.csv):

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

Products (products.csv):

| $j$ | ProductName            | Value | Weight |
|-----|------------------------|-------|--------|
| 1   | Smartphone             | 200   | 1      |
| 2   | Laptop                 | 1500  | 5      |
| 3   | Headphones             | 100   | 0.5    |
| 4   | Camera                 | 800   | 2      |
| 5   | Smartwatch             | 250   | 0.3    |
| 6   | Tablet                 | 600   | 1.5    |
| 7   | Bluetooth Speaker      | 150   | 1      |
| 8   | Keyboard               | 80    | 0.8    |
| 9   | Mouse                  | 50    | 0.2    |
| 10  | Monitor                | 300   | 3      |
| 11  | Printer                | 400   | 4      |
| 12  | External Hard Drive    | 120   | 0.5    |
| 13  | Router                 | 60    | 0.3    |
| 14  | Power Bank             | 40    | 0.4    |
| 15  | Memory Card            | 30    | 0.05   |
| 16  | USB Flash Drive        | 25    | 0.02   |
| 17  | Smart Home Hub         | 100   | 0.6    |
| 18  | Gaming Console         | 500   | 4      |
| 19  | Fitness Tracker        | 90    | 0.2    |
| 20  | E-Reader               | 180   | 0.5    |

Mathematical Model:

Objective:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

Subject to:

1. Display capacity constraints (for each display $i$):
$$
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
$$

Explicitly, for each $i$ (using the data above):

- For $i=1$: $\sum_{j=1}^{20} w_j x_{1j} \leq 5$
- For $i=2$: $\sum_{j=1}^{20} w_j x_{2j} \leq 7$
- For $i=3$: $\sum_{j=1}^{20} w_j x_{3j} \leq 6$
- For $i=4$: $\sum_{j=1}^{20} w_j x_{4j} \leq 8$
- For $i=5$: $\sum_{j=1}^{20} w_j x_{5j} \leq 5.5$
- For $i=6$: $\sum_{j=1}^{20} w_j x_{6j} \leq 9$
- For $i=7$: $\sum_{j=1}^{20} w_j x_{7j} \leq 6.5$
- For $i=8$: $\sum_{j=1}^{20} w_j x_{8j} \leq 7.5$
- For $i=9$: $\sum_{j=1}^{20} w_j x_{9j} \leq 8.2$
- For $i=10$: $\sum_{j=1}^{20} w_j x_{10j} \leq 5.7$

2. Minimum total quantity of the first product ("Smartphone") across all displays:
$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

3. Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

Where:
- $v_j$ and $w_j$ are as given in the table above for each product $j$.
- $C_i$ is as given in the table above for each display $i$.

All identifiers and coefficients are preserved in source order.