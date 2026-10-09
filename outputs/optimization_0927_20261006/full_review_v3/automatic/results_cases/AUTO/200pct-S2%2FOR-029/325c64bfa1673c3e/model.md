Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

Let $I$ be the set of displays (shelves), indexed by $i$ with ShelfID as below.

Let $J$ be the set of products, indexed by $j$ with ProductName as below.

Let $v_j$ be the value of product $j$.

Let $w_j$ be the weight of product $j$.

Let $C_i$ be the capacity of display $i$.

The first product in source order is "Smartphone".

---

**Sets and Parameters (in source order):**

Displays (from capacity.csv):

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

Products (from products.csv):

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

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{\text{Smartphone}, \text{Laptop}, \ldots, \text{E-Reader}\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

where $v_j$ is the value of product $j$ as listed above.

**Constraints:**

1. **Display Capacity Constraints (for each display $i$):**

$$
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
$$

where $w_j$ is the weight of product $j$ and $C_i$ is the capacity of display $i$ as listed above.

2. **Minimum Smartphone Allocation (across all displays):**

$$
\sum_{i=1}^{10} x_{i,\ \text{Smartphone}} \geq 5
$$

3. **Nonnegativity and Integrality:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**All identifiers and coefficients are as retrieved and in source order.**