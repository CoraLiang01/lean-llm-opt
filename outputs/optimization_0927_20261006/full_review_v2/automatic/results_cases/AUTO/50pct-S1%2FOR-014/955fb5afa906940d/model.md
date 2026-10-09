Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$. All $x_{ij}$ are integer and $\geq 0$.

**Sets and Indices:**
- $i$ indexes shelves, with ShelfID from capacity.csv.
- $j$ indexes products, with ProductName from products.csv.

**Parameters:**
- $c_i$ = Capacity of shelf $i$ (from capacity.csv, column "Capacity")
- $v_j$ = Value of product $j$ (from products.csv, column "Value")
- $w_j$ = Weight of product $j$ (from products.csv, column "Weight")

**Data (in source order):**

*Shelves (capacity.csv):*

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

*Products (products.csv):*

| ProductName             | Value | Weight |
|-------------------------|-------|--------|
| Smartphone              | 200   | 1.0    |
| Laptop                  | 1500  | 5.0    |
| Headphones              | 100   | 0.5    |
| Camera                  | 800   | 2.0    |
| Smartwatch              | 250   | 0.3    |
| Tablet                  | 600   | 1.5    |
| Bluetooth Speaker       | 150   | 1.0    |
| Keyboard                | 80    | 0.8    |
| Mouse                   | 50    | 0.2    |
| Monitor                 | 300   | 3.0    |
| Printer                 | 400   | 4.0    |
| External Hard Drive     | 120   | 0.5    |
| Router                  | 60    | 0.3    |
| Power Bank              | 40    | 0.4    |
| Memory Card             | 30    | 0.05   |
| USB Flash Drive         | 25    | 0.02   |
| Smart Home Hub          | 100   | 0.6    |
| Gaming Console          | 500   | 4.0    |
| Fitness Tracker         | 90    | 0.2    |
| E-Reader                | 180   | 0.5    |

---

### Mathematical Model

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall \text{ ShelfID } i, \text{ ProductName } j
$$

**Objective:**
$$
\max \sum_{i \in \{\text{1,...,10}\}} \sum_{j \in \{\text{all products}\}} v_j \cdot x_{ij}
$$

**Shelf Capacity Constraints:**
$$
\sum_{j} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{\text{1,...,10}\}
$$

**Variable Domains:**
$$
x_{ij} \in \{0,1,2,\ldots\} \qquad \forall i, j
$$

---

**Where:**

- $i$ runs over ShelfID: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- $j$ runs over ProductName: Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader
- $c_i$ is the Capacity for shelf $i$ (see table above)
- $v_j$ is the Value for product $j$ (see table above)
- $w_j$ is the Weight for product $j$ (see table above)

All $x_{ij}$ are nonnegative integers.