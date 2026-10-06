Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

**Parameters:**

- Let $i$ index shelves (from capacity.csv, column ShelfID): $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- Let $j$ index products (from products.csv, in source order): $j \in \{1,2,\ldots,20\}$
- Let $c_i$ be the capacity of shelf $i$ (from capacity.csv, column Capacity)
- Let $v_j$ be the value of product $j$ (from products.csv, column Value)
- Let $w_j$ be the weight of product $j$ (from products.csv, column Weight)

**Data:**

From capacity.csv (in source order):

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

From products.csv (in source order):

| j | ProductName           | Value | Weight |
|---|-----------------------|-------|--------|
| 1 | Smartphone            | 200   | 1      |
| 2 | Laptop                | 1500  | 5      |
| 3 | Headphones            | 100   | 0.5    |
| 4 | Camera                | 800   | 2      |
| 5 | Smartwatch            | 250   | 0.3    |
| 6 | Tablet                | 600   | 1.5    |
| 7 | Bluetooth Speaker     | 150   | 1      |
| 8 | Keyboard              | 80    | 0.8    |
| 9 | Mouse                 | 50    | 0.2    |
|10 | Monitor               | 300   | 3      |
|11 | Printer               | 400   | 4      |
|12 | External Hard Drive   | 120   | 0.5    |
|13 | Router                | 60    | 0.3    |
|14 | Power Bank            | 40    | 0.4    |
|15 | Memory Card           | 30    | 0.05   |
|16 | USB Flash Drive       | 25    | 0.02   |
|17 | Smart Home Hub        | 100   | 0.6    |
|18 | Gaming Console        | 500   | 4      |
|19 | Fitness Tracker       | 90    | 0.2    |
|20 | E-Reader              | 180   | 0.5    |

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]

**Subject to:**

1. **Shelf capacity constraints (for each shelf $i$):**
\[
\sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

2. **Minimum allocation of the first product (Smartphone) across all shelves:**
\[
\sum_{i=1}^{10} x_{i1} \geq 5
\]

3. **Nonnegativity and integrality:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

**Where:**

- $v_j$ and $w_j$ are as given in the table above for each product $j$.
- $c_i$ is as given in the table above for each shelf $i$.

**Decision variables:**

- $x_{ij}$: Number of units of product $j$ placed on shelf $i$ (nonnegative integer).