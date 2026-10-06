**Sets and Indices:**

- Let $i$ index the shelves (displays), with $i \in \{1,2,3,4,5,6,7,8,9,10\}$, corresponding to ShelfID from capacity.csv.
- Let $j$ index the products, with $j \in \{1,2,\ldots,20\}$, corresponding to the order of products in products.csv.

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

Let $v_j$ be the Value of product $j$.

Let $w_j$ be the Weight of product $j$.

Let $C_i$ be the Capacity of shelf $i$.

**Decision Variables:**

- $x_{ij}$ = number of units of product $j$ placed on shelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

**Subject to:**

1. **Shelf Capacity Constraints:** For each shelf $i$,
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
\]

2. **Minimum Quantity of First Product Across All Shelves:**
\[
\sum_{i=1}^{10} x_{i1} \geq 5
\]

3. **Nonnegativity and Integrality:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
\]

---

**Parameter Table (for reference):**

- $C_1 = 5$, $C_2 = 7$, $C_3 = 6$, $C_4 = 8$, $C_5 = 5.5$, $C_6 = 9$, $C_7 = 6.5$, $C_8 = 7.5$, $C_9 = 8.2$, $C_{10} = 5.7$
- $(v_1, w_1) = (200, 1)$, $(v_2, w_2) = (1500, 5)$, $(v_3, w_3) = (100, 0.5)$, $(v_4, w_4) = (800, 2)$, $(v_5, w_5) = (250, 0.3)$, $(v_6, w_6) = (600, 1.5)$, $(v_7, w_7) = (150, 1)$, $(v_8, w_8) = (80, 0.8)$, $(v_9, w_9) = (50, 0.2)$, $(v_{10}, w_{10}) = (300, 3)$, $(v_{11}, w_{11}) = (400, 4)$, $(v_{12}, w_{12}) = (120, 0.5)$, $(v_{13}, w_{13}) = (60, 0.3)$, $(v_{14}, w_{14}) = (40, 0.4)$, $(v_{15}, w_{15}) = (30, 0.05)$, $(v_{16}, w_{16}) = (25, 0.02)$, $(v_{17}, w_{17}) = (100, 0.6)$, $(v_{18}, w_{18}) = (500, 4)$, $(v_{19}, w_{19}) = (90, 0.2)$, $(v_{20}, w_{20}) = (180, 0.5)$

---

**Summary:**

Maximize total value of products allocated to shelves, subject to each shelf's weight capacity, and ensuring at least 5 units of the first product ("Smartphone") are placed across all shelves. All variables are nonnegative integers.