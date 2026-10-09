Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Let $I$ be the set of shelves (from capacity.csv, in source order):
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$, with ShelfID as below.

Let $J$ be the set of products (from products.csv, in source order):
- $j \in \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$, with ProductName as below.

Let $v_j$ be the value of product $j$ (from Value column).
Let $w_j$ be the weight of product $j$ (from Weight column).
Let $C_i$ be the capacity of shelf $i$ (from Capacity column).

The model is:

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Subject to:**

1. **Shelf capacity constraints (for each shelf $i$):**
\[
\sum_{j \in J} w_j \, x_{ij} \leq C_i \qquad \forall i \in I
\]

2. **Minimum total quantity of the first product ("Smartphone") across all shelves:**
\[
\sum_{i \in I} x_{i,\text{Smartphone}} \geq 5
\]

3. **Nonnegativity and integrality:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

**Parameter Data (in source order):**

- **Shelves (from capacity.csv):**

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

- **Products (from products.csv, in source order):**

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

---

**Decision variables:**
\[
x_{ij} = \text{number of units of product $j$ placed on shelf $i$}, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**All parameters and constraints are as above, using the exact identifiers and values from the retrieved data.**