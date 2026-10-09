Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $i$ indexes ShelfID from the capacity.csv file and $j$ indexes ProductName from the products.csv file. All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $S$ = set of shelves (from capacity.csv):  
  $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$

- $P$ = set of products (from products.csv):  
  $P = \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$

- $v_j$ = value of one unit of product $j$ (from Value column in products.csv)
- $w_j$ = weight of one unit of product $j$ (from Weight column in products.csv)
- $C_i$ = capacity of shelf $i$ (from Capacity column in capacity.csv)

**Data:**

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

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

**Subject to:**

- Shelf capacity constraints (for each shelf $i$):
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in S
\]

- Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
\]

**Where:**

- $x_{ij}$ = number of units of product $j$ placed on shelf $i$ (integer, $\geq 0$)
- $v_j$ = value of product $j$ (see table above)
- $w_j$ = weight of product $j$ (see table above)
- $C_i$ = capacity of shelf $i$ (see table above)

**All identifiers and coefficients are as retrieved and preserved in original order.**