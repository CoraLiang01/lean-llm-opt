Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $i$ indexes ShelfID from the capacity.csv file and $j$ indexes ProductName from the products.csv file. All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight of product $j$ (from products.csv)
- $C_i$ = Capacity of shelf $i$ (from capacity.csv)

**Sets:**

- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (ShelfID)
- $j \in \{$
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
$\}$ (ProductName)

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

**Subject to:**

For each shelf $i$ (using ShelfID and Capacity from capacity.csv):

\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

**Variable domains:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

**Parameter Data (in source order):**

*Shelves (from capacity.csv):*

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

*Products (from products.csv):*

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

---

**Summary:**

- Decision variables: $x_{ij}$ = number of units of product $j$ on shelf $i$, integer, $\geq 0$
- Objective: maximize total value across all shelves and products
- Constraints: for each shelf, total weight of products allocated does not exceed that shelf's capacity

All data, identifiers, and coefficients are preserved in the formulation above.