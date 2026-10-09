Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (ShelfID from capacity.csv) and $j \in \{1,2,\ldots,20\}$ (products in the order listed below).

Let $v_j$ be the value of product $j$, and $w_j$ be the weight of product $j$ (from products.csv). Let $C_i$ be the capacity of shelf $i$ (from capacity.csv).

#### Product Index Mapping (in source order):

| $j$ | ProductName             | $v_j$ | $w_j$  |
|-----|------------------------|-------|--------|
| 1   | Smartphone             | 200   | 1.0    |
| 2   | Laptop                 | 1500  | 5.0    |
| 3   | Headphones             | 100   | 0.5    |
| 4   | Camera                 | 800   | 2.0    |
| 5   | Smartwatch             | 250   | 0.3    |
| 6   | Tablet                 | 600   | 1.5    |
| 7   | Bluetooth Speaker      | 150   | 1.0    |
| 8   | Keyboard               | 80    | 0.8    |
| 9   | Mouse                  | 50    | 0.2    |
| 10  | Monitor                | 300   | 3.0    |
| 11  | Printer                | 400   | 4.0    |
| 12  | External Hard Drive    | 120   | 0.5    |
| 13  | Router                 | 60    | 0.3    |
| 14  | Power Bank             | 40    | 0.4    |
| 15  | Memory Card            | 30    | 0.05   |
| 16  | USB Flash Drive        | 25    | 0.02   |
| 17  | Smart Home Hub         | 100   | 0.6    |
| 18  | Gaming Console         | 500   | 4.0    |
| 19  | Fitness Tracker        | 90    | 0.2    |
| 20  | E-Reader               | 180   | 0.5    |

#### Shelf Index Mapping (in source order):

| $i$ | ShelfID | $C_i$ |
|-----|---------|-------|
| 1   | 1       | 5.0   |
| 2   | 2       | 7.0   |
| 3   | 3       | 6.0   |
| 4   | 4       | 8.0   |
| 5   | 5       | 5.5   |
| 6   | 6       | 9.0   |
| 7   | 7       | 6.5   |
| 8   | 8       | 7.5   |
| 9   | 9       | 8.2   |
| 10  | 10      | 5.7   |

---

### Mathematical Model

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

**Subject to:**

1. **Shelf Capacity Constraints (for each shelf $i$):**
   $$
   \sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
   $$
   That is, for each shelf:
   - Shelf 1: $\sum_{j=1}^{20} w_j x_{1j} \leq 5.0$
   - Shelf 2: $\sum_{j=1}^{20} w_j x_{2j} \leq 7.0$
   - Shelf 3: $\sum_{j=1}^{20} w_j x_{3j} \leq 6.0$
   - Shelf 4: $\sum_{j=1}^{20} w_j x_{4j} \leq 8.0$
   - Shelf 5: $\sum_{j=1}^{20} w_j x_{5j} \leq 5.5$
   - Shelf 6: $\sum_{j=1}^{20} w_j x_{6j} \leq 9.0$
   - Shelf 7: $\sum_{j=1}^{20} w_j x_{7j} \leq 6.5$
   - Shelf 8: $\sum_{j=1}^{20} w_j x_{8j} \leq 7.5$
   - Shelf 9: $\sum_{j=1}^{20} w_j x_{9j} \leq 8.2$
   - Shelf 10: $\sum_{j=1}^{20} w_j x_{10j} \leq 5.7$

2. **Minimum Total Quantity of First Product (Smartphone):**
   $$
   \sum_{i=1}^{10} x_{i1} \geq 5
   $$

3. **Nonnegativity and Integrality:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
   $$

---

**Parameter Tables (from source order):**

*Shelf Capacities:*

| ShelfID | $C_i$ |
|---------|-------|
| 1       | 5.0   |
| 2       | 7.0   |
| 3       | 6.0   |
| 4       | 8.0   |
| 5       | 5.5   |
| 6       | 9.0   |
| 7       | 6.5   |
| 8       | 7.5   |
| 9       | 8.2   |
| 10      | 5.7   |

*Product Values and Weights:*

| $j$ | ProductName             | $v_j$ | $w_j$  |
|-----|------------------------|-------|--------|
| 1   | Smartphone             | 200   | 1.0    |
| 2   | Laptop                 | 1500  | 5.0    |
| 3   | Headphones             | 100   | 0.5    |
| 4   | Camera                 | 800   | 2.0    |
| 5   | Smartwatch             | 250   | 0.3    |
| 6   | Tablet                 | 600   | 1.5    |
| 7   | Bluetooth Speaker      | 150   | 1.0    |
| 8   | Keyboard               | 80    | 0.8    |
| 9   | Mouse                  | 50    | 0.2    |
| 10  | Monitor                | 300   | 3.0    |
| 11  | Printer                | 400   | 4.0    |
| 12  | External Hard Drive    | 120   | 0.5    |
| 13  | Router                 | 60    | 0.3    |
| 14  | Power Bank             | 40    | 0.4    |
| 15  | Memory Card            | 30    | 0.05   |
| 16  | USB Flash Drive        | 25    | 0.02   |
| 17  | Smart Home Hub         | 100   | 0.6    |
| 18  | Gaming Console         | 500   | 4.0    |
| 19  | Fitness Tracker        | 90    | 0.2    |
| 20  | E-Reader               | 180   | 0.5    |