Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where:

- $i$ indexes ShelfID from 1 to 10 (from capacity.csv, in order)
- $j$ indexes ProductName from 1 to 20 (from products.csv, in order)

Let $c_i$ be the Capacity of shelf $i$ (from the Capacity column of capacity.csv).
Let $v_j$ be the Value of product $j$ (from the Value column of products.csv).
Let $w_j$ be the Weight of product $j$ (from the Weight column of products.csv).

The first product (row 0 of products.csv) is "Smartphone".

---

**Parameters:**

From capacity.csv (in order):

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

From products.csv (in order):

| $j$ | ProductName           | Value | Weight |
|-----|-----------------------|-------|--------|
| 1   | Smartphone            | 200   | 1      |
| 2   | Laptop                | 1500  | 5      |
| 3   | Headphones            | 100   | 0.5    |
| 4   | Camera                | 800   | 2      |
| 5   | Smartwatch            | 250   | 0.3    |
| 6   | Tablet                | 600   | 1.5    |
| 7   | Bluetooth Speaker     | 150   | 1      |
| 8   | Keyboard              | 80    | 0.8    |
| 9   | Mouse                 | 50    | 0.2    |
| 10  | Monitor               | 300   | 3      |
| 11  | Printer               | 400   | 4      |
| 12  | External Hard Drive   | 120   | 0.5    |
| 13  | Router                | 60    | 0.3    |
| 14  | Power Bank            | 40    | 0.4    |
| 15  | Memory Card           | 30    | 0.05   |
| 16  | USB Flash Drive       | 25    | 0.02   |
| 17  | Smart Home Hub        | 100   | 0.6    |
| 18  | Gaming Console        | 500   | 4      |
| 19  | Fitness Tracker       | 90    | 0.2    |
| 20  | E-Reader              | 180   | 0.5    |

---

**Mathematical Model:**

Objective:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}
$$

Subject to:

1. **Shelf Capacity Constraints** (for each shelf $i$):
$$
\sum_{j=1}^{20} w_j \, x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10
$$

2. **Minimum Smartphone Allocation** (across all shelves):
$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

3. **Nonnegativity and Integrality:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
$$

---

**Where:**

- $x_{ij}$: Number of units of product $j$ placed on shelf $i$
- $v_j$: Value of product $j$ (see table above)
- $w_j$: Weight of product $j$ (see table above)
- $c_i$: Capacity of shelf $i$ (see table above)

All indices, coefficients, and constraints are as retrieved and in original order.