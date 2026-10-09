Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

Let $i$ index the shelves (displays) using the ShelfID from capacity.csv:
$$
i \in \{1,2,3,4,5,6,7,8,9,10\}
$$

Let $j$ index the products in the order they appear in products.csv:
$$
j \in \{1,2,\ldots,20\}
$$

Let $v_j$ be the value of product $j$ (from the "Value" column in products.csv).

Let $w_j$ be the weight of product $j$ (from the "Weight" column in products.csv).

Let $C_i$ be the capacity of shelf $i$ (from the "Capacity" column in capacity.csv).

The first product is "Smartphone" (the first row of products.csv, $j=1$).

---

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

**Subject to:**

1. **Shelf capacity constraints:** For each shelf $i$,
$$
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

2. **Minimum total quantity of first product:**
$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

3. **Nonnegativity and integrality:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\; j \in \{1,\ldots,20\}
$$

---

**Parameter Data (in source order):**

*Shelves (from capacity.csv):*

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

*Products (from products.csv, in order):*

| $j$ | ProductName             | Value | Weight |
|-----|------------------------|-------|--------|
| 1   | Smartphone             | 200   | 1      |
| 2   | Laptop                 | 1500  | 5      |
| 3   | Headphones             | 100   | 0.5    |
| 4   | Camera                 | 800   | 2      |
| 5   | Smartwatch             | 250   | 0.3    |
| 6   | Tablet                 | 600   | 1.5    |
| 7   | Bluetooth Speaker      | 150   | 1      |
| 8   | Keyboard               | 80    | 0.8    |
| 9   | Mouse                  | 50    | 0.2    |
| 10  | Monitor                | 300   | 3      |
| 11  | Printer                | 400   | 4      |
| 12  | External Hard Drive    | 120   | 0.5    |
| 13  | Router                 | 60    | 0.3    |
| 14  | Power Bank             | 40    | 0.4    |
| 15  | Memory Card            | 30    | 0.05   |
| 16  | USB Flash Drive        | 25    | 0.02   |
| 17  | Smart Home Hub         | 100   | 0.6    |
| 18  | Gaming Console         | 500   | 4      |
| 19  | Fitness Tracker        | 90    | 0.2    |
| 20  | E-Reader               | 180   | 0.5    |

---

**Summary of Model:**

Maximize total value of all products placed on all shelves, subject to:
- The total weight of products on each shelf does not exceed its capacity.
- At least 5 units of "Smartphone" (product 1) are placed in total.
- All $x_{ij}$ are nonnegative integers.