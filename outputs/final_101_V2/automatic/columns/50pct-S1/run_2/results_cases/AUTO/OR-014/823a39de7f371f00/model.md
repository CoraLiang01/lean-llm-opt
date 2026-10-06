Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $i$ indexes ShelfID from capacity.csv and $j$ indexes ProductName from products.csv. All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $V_j$: Value of product $j$ (from products.csv)
- $W_j$: Weight of product $j$ (from products.csv)
- $C_i$: Capacity of shelf $i$ (from capacity.csv)

**Sets:**

- $I$: Set of shelves (ShelfID from capacity.csv): $\{1,2,3,4,5,6,7,8,9,10\}$
- $J$: Set of products (ProductName from products.csv): 
  - Smartphone
  - Laptop
  - Headphones
  - Camera
  - Smartwatch
  - Tablet
  - Bluetooth Speaker
  - Keyboard
  - Mouse
  - Monitor
  - Printer
  - External Hard Drive
  - Router
  - Power Bank
  - Memory Card
  - USB Flash Drive
  - Smart Home Hub
  - Gaming Console
  - Fitness Tracker
  - E-Reader

**Data:**

- Shelf capacities:

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

- Product values and weights:

| ProductName           | Value | Weight |
|----------------------|-------|--------|
| Smartphone           | 200   | 1.0    |
| Laptop               | 1500  | 5.0    |
| Headphones           | 100   | 0.5    |
| Camera               | 800   | 2.0    |
| Smartwatch           | 250   | 0.3    |
| Tablet               | 600   | 1.5    |
| Bluetooth Speaker    | 150   | 1.0    |
| Keyboard             | 80    | 0.8    |
| Mouse                | 50    | 0.2    |
| Monitor              | 300   | 3.0    |
| Printer              | 400   | 4.0    |
| External Hard Drive  | 120   | 0.5    |
| Router               | 60    | 0.3    |
| Power Bank           | 40    | 0.4    |
| Memory Card          | 30    | 0.05   |
| USB Flash Drive      | 25    | 0.02   |
| Smart Home Hub       | 100   | 0.6    |
| Gaming Console       | 500   | 4.0    |
| Fitness Tracker      | 90    | 0.2    |
| E-Reader             | 180   | 0.5    |

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} V_j \cdot x_{ij}
\]

**Subject to:**

For each shelf $i \in I$:
\[
\sum_{j \in J} W_j \cdot x_{ij} \leq C_i
\]

For all $i \in I$, $j \in J$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Where:**

- $x_{ij}$: Number of units of product $j$ placed on shelf $i$ (integer, $\geq 0$)
- $V_j$: Value of product $j$ (see table above)
- $W_j$: Weight of product $j$ (see table above)
- $C_i$: Capacity of shelf $i$ (see table above)
- $I$: $\{1,2,3,4,5,6,7,8,9,10\}$
- $J$: List of 20 products as above

All data and identifiers are preserved in original order.