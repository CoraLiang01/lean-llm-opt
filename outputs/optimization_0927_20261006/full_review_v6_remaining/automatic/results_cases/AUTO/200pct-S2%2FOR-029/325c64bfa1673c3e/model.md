Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

**Sets and Indices:**
- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (ShelfID from capacity.csv)
- $j \in \{\text{Smartphone}, \text{Laptop}, \text{Headphones}, \text{Camera}, \text{Smartwatch}, \text{Tablet}, \text{Bluetooth Speaker}, \text{Keyboard}, \text{Mouse}, \text{Monitor}, \text{Printer}, \text{External Hard Drive}, \text{Router}, \text{Power Bank}, \text{Memory Card}, \text{USB Flash Drive}, \text{Smart Home Hub}, \text{Gaming Console}, \text{Fitness Tracker}, \text{E-Reader}\}$ (ProductName from products.csv)

**Parameters:**

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

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all shelves $i$ and products $j$

---

### Mathematical Model

**Objective:**
\[
\max \sum_{i \in \{\text{1},\ldots,\text{10}\}} \sum_{j \in \{\text{Smartphone}, \ldots, \text{E-Reader}\}} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of product $j$.

**Constraints:**

1. **Shelf Capacity Constraints (for each shelf $i$):**
   \[
   \sum_{j} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{\text{1},\ldots,\text{10}\}
   \]
   where $w_j$ is the Weight of product $j$, $c_i$ is the Capacity of shelf $i$.

   Explicitly, for each shelf:

   - Shelf 1: $\sum_{j} w_j x_{1j} \leq 5$
   - Shelf 2: $\sum_{j} w_j x_{2j} \leq 7$
   - Shelf 3: $\sum_{j} w_j x_{3j} \leq 6$
   - Shelf 4: $\sum_{j} w_j x_{4j} \leq 8$
   - Shelf 5: $\sum_{j} w_j x_{5j} \leq 5.5$
   - Shelf 6: $\sum_{j} w_j x_{6j} \leq 9$
   - Shelf 7: $\sum_{j} w_j x_{7j} \leq 6.5$
   - Shelf 8: $\sum_{j} w_j x_{8j} \leq 7.5$
   - Shelf 9: $\sum_{j} w_j x_{9j} \leq 8.2$
   - Shelf 10: $\sum_{j} w_j x_{10j} \leq 5.7$

2. **Minimum Total Quantity for First Product ("Smartphone"):**
   \[
   \sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5
   \]

3. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
   \]

---

**Parameter Table (for reference):**

| $j$ (ProductName)         | $v_j$ (Value) | $w_j$ (Weight) |
|---------------------------|---------------|---------------|
| Smartphone                | 200           | 1             |
| Laptop                    | 1500          | 5             |
| Headphones                | 100           | 0.5           |
| Camera                    | 800           | 2             |
| Smartwatch                | 250           | 0.3           |
| Tablet                    | 600           | 1.5           |
| Bluetooth Speaker         | 150           | 1             |
| Keyboard                  | 80            | 0.8           |
| Mouse                     | 50            | 0.2           |
| Monitor                   | 300           | 3             |
| Printer                   | 400           | 4             |
| External Hard Drive       | 120           | 0.5           |
| Router                    | 60            | 0.3           |
| Power Bank                | 40            | 0.4           |
| Memory Card               | 30            | 0.05          |
| USB Flash Drive           | 25            | 0.02          |
| Smart Home Hub            | 100           | 0.6           |
| Gaming Console            | 500           | 4             |
| Fitness Tracker           | 90            | 0.2           |
| E-Reader                  | 180           | 0.5           |

| $i$ (ShelfID) | $c_i$ (Capacity) |
|---------------|------------------|
| 1             | 5                |
| 2             | 7                |
| 3             | 6                |
| 4             | 8                |
| 5             | 5.5              |
| 6             | 9                |
| 7             | 6.5              |
| 8             | 7.5              |
| 9             | 8.2              |
| 10            | 5.7              |

---

**Summary:**

Maximize total value of all products placed on all shelves, subject to:
- Each shelf's total product weight not exceeding its capacity,
- At least 5 units of "Smartphone" placed in total,
- All $x_{ij}$ are nonnegative integers.