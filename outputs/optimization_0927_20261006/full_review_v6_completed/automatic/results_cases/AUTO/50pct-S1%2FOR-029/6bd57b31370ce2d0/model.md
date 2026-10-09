Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i$ indexes ShelfID from 1 to 10 and $j$ indexes the 20 products in the order given below.

**Parameters:**

- Displays (shelves): $i \in \{1,2,3,4,5,6,7,8,9,10\}$, with capacities $C_i$ as follows:

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

- Products $j \in \{1,2,\ldots,20\}$, in the following order, with values $v_j$ and weights $w_j$:

| $j$ | ProductName           | Value | Weight |
|-----|-----------------------|-------|--------|
| 1   | Smartphone           | 200   | 1      |
| 2   | Laptop               | 1500  | 5      |
| 3   | Headphones           | 100   | 0.5    |
| 4   | Camera               | 800   | 2      |
| 5   | Smartwatch           | 250   | 0.3    |
| 6   | Tablet               | 600   | 1.5    |
| 7   | Bluetooth Speaker    | 150   | 1      |
| 8   | Keyboard             | 80    | 0.8    |
| 9   | Mouse                | 50    | 0.2    |
| 10  | Monitor              | 300   | 3      |
| 11  | Printer              | 400   | 4      |
| 12  | External Hard Drive  | 120   | 0.5    |
| 13  | Router               | 60    | 0.3    |
| 14  | Power Bank           | 40    | 0.4    |
| 15  | Memory Card          | 30    | 0.05   |
| 16  | USB Flash Drive      | 25    | 0.02   |
| 17  | Smart Home Hub       | 100   | 0.6    |
| 18  | Gaming Console       | 500   | 4      |
| 19  | Fitness Tracker      | 90    | 0.2    |
| 20  | E-Reader             | 180   | 0.5    |

**Decision Variables:**

- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i=1,\ldots,10$, $j=1,\ldots,20$

---

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}
$$

---

**Constraints:**

1. **Shelf Capacity Constraints:** For each shelf $i$,
   $$
   \sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \qquad \forall i=1,\ldots,10
   $$
   where $C_i$ is the Capacity for ShelfID $i$.

2. **Minimum Total Quantity of First Product:**
   $$
   \sum_{i=1}^{10} x_{i1} \geq 5
   $$
   (where $x_{i1}$ corresponds to "Smartphone" on shelf $i$)

3. **Nonnegativity and Integrality:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10,\; j=1,\ldots,20
   $$

---

**All parameters and indices are as given in the retrieved data above.**