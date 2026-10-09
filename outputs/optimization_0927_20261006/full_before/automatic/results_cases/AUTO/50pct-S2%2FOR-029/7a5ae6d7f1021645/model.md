Let $x_{ij}$ be the number of units of product $j$ placed on display $i$, where $i$ indexes ShelfID from capacity.csv and $j$ indexes ProductName from products.csv. All $x_{ij}$ are nonnegative integers.

**Sets and Indices:**
- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (ShelfID)
- $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$ (ProductName)

**Parameters:**
- $v_j$ = Value of product $j$ (see table below)
- $w_j$ = Weight of product $j$ (see table below)
- $C_i$ = Capacity of shelf $i$ (see table below)

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

| ProductName           | Value | Weight |
|---------------------- |-------|--------|
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

**Mathematical Model:**

Objective:
\[
\max \sum_{i \in \{\text{1},\ldots,\text{10}\}} \sum_{j \in \{\text{Smartphone},\ldots,\text{E-Reader}\}} v_j x_{ij}
\]

Subject to:

Capacity constraints for each shelf:
\[
\sum_{j} w_j x_{ij} \leq C_i \qquad \forall i \in \{\text{1},\ldots,\text{10}\}
\]

Minimum allocation for the first product (Smartphone):
\[
\sum_{i} x_{i,\text{Smartphone}} \geq 5
\]

Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]