Let $x_{ij}$ be the number of units of product $j$ (ProductName from products.csv) to be placed on shelf $i$ (ShelfID from capacity.csv). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $V_j$: Value of one unit of product $j$ (from "Value" in products.csv)
- $w_j$: Weight of one unit of product $j$ (from "Weight" in products.csv)
- $C_i$: Capacity of shelf $i$ (from "Capacity" in capacity.csv)

**Sets:**

- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (ShelfID)
- $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$ (ProductName)

**Data:**

From capacity.csv (in source order):

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

From products.csv (in source order):

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Smartphone          | 200   | 1.0    |
| Laptop              | 1500  | 5.0    |
| Headphones          | 100   | 0.5    |
| Camera              | 800   | 2.0    |
| Smartwatch          | 250   | 0.3    |
| Tablet              | 600   | 1.5    |
| Bluetooth Speaker   | 150   | 1.0    |
| Keyboard            | 80    | 0.8    |
| Mouse               | 50    | 0.2    |
| Monitor             | 300   | 3.0    |
| Printer             | 400   | 4.0    |
| External Hard Drive | 120   | 0.5    |
| Router              | 60    | 0.3    |
| Power Bank          | 40    | 0.4    |
| Memory Card         | 30    | 0.05   |
| USB Flash Drive     | 25    | 0.02   |
| Smart Home Hub      | 100   | 0.6    |
| Gaming Console      | 500   | 4.0    |
| Fitness Tracker     | 90    | 0.2    |
| E-Reader            | 180   | 0.5    |

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} = \text{number of units of product } j \text{ placed on shelf } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

**Objective:**

$$
\max \sum_{i \in \{\text{1},\ldots,\text{10}\}} \sum_{j \in \{\text{all products above}\}} V_j \cdot x_{ij}
$$

**Subject to:**

For each shelf $i$ (ShelfID from 1 to 10):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{1},\ldots,\text{10}\}
$$

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Where:**

- $V_j$ and $w_j$ are as given in the table above for each product $j$.
- $C_i$ is as given in the table above for each shelf $i$.

**All data and identifiers are preserved in original order as retrieved.**