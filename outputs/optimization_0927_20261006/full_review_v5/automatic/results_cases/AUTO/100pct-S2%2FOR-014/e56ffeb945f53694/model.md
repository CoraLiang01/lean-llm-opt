Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all shelves $i$ and products $j$.

**Sets:**
- $I$: Set of shelves, indexed by ShelfID.
- $J$: Set of products, indexed by ProductName.

**Parameters:**
- $v_j$: Value of one unit of product $j$ (from Value column).
- $w_j$: Weight of one unit of product $j$ (from Weight column).
- $C_i$: Capacity of shelf $i$ (from Capacity column).

---

### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

For each shelf $i \in I$:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i
\]

For all $i \in I$, $j \in J$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

### Parameter Tables

**Shelves (from capacity.csv, in source order):**

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

**Products (from products.csv, in source order):**

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

**Decision Variables:**
- $x_{ij}$: Number of units of product $j$ placed on shelf $i$, integer and nonnegative.

---

**Summary:**

Maximize total value of products allocated to shelves, subject to each shelf's weight capacity, with integer allocation decisions for each product-shelf pair. All data and identifiers are preserved as retrieved.