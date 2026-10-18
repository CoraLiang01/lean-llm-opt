**Sets and Indices:**

- Let $I$ be the set of shelves, indexed by $i$ (with ShelfID from 1 to 10).
- Let $J$ be the set of products, indexed by $j$ (with ProductName from 1 to 20).

**Parameters:**

- $C_i$: Capacity of shelf $i$ (from "capacity.csv").
- $v_j$: Value of product $j$ (from "products.csv").
- $w_j$: Weight of product $j$ (from "products.csv").

**Decision Variables:**

- $x_{ij}$: Number of units of product $j$ to be placed on shelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

---

**Data:**

From "capacity.csv":

| ShelfID | Capacity |
|---------|----------|
| 1       | 500      |
| 2       | 700      |
| 3       | 600      |
| 4       | 800      |
| 5       | 550      |
| 6       | 900      |
| 7       | 650      |
| 8       | 750      |
| 9       | 820      |
| 10      | 570      |

From "products.csv":

| ProductName | Value | Weight |
|-------------|-------|--------|
| 1           | 50    | 10     |
| 2           | 70    | 20     |
| 3           | 30    | 5      |
| 4           | 60    | 15     |
| 5           | 80    | 25     |
| 6           | 90    | 30     |
| 7           | 40    | 12     |
| 8           | 100   | 35     |
| 9           | 55    | 10     |
| 10          | 75    | 20     |
| 11          | 65    | 18     |
| 12          | 95    | 28     |
| 13          | 45    | 8      |
| 14          | 85    | 22     |
| 15          | 70    | 25     |
| 16          | 110   | 40     |
| 17          | 50    | 14     |
| 18          | 60    | 16     |
| 19          | 120   | 50     |
| 20          | 100   | 30     |

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]
where $v_j$ is the value of product $j$ as given above.

**Subject to:**

For each shelf $i$ (ShelfID $i$):

\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
\]
where $w_j$ is the weight of product $j$ and $C_i$ is the capacity of shelf $i$ as given above.

**Variable Domains:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
\]

---

**Explicit Data Mapping:**

- $C_1 = 500$, $C_2 = 700$, $C_3 = 600$, $C_4 = 800$, $C_5 = 550$, $C_6 = 900$, $C_7 = 650$, $C_8 = 750$, $C_9 = 820$, $C_{10} = 570$
- $(v_1, w_1) = (50, 10)$, $(v_2, w_2) = (70, 20)$, $(v_3, w_3) = (30, 5)$, $(v_4, w_4) = (60, 15)$, $(v_5, w_5) = (80, 25)$, $(v_6, w_6) = (90, 30)$, $(v_7, w_7) = (40, 12)$, $(v_8, w_8) = (100, 35)$, $(v_9, w_9) = (55, 10)$, $(v_{10}, w_{10}) = (75, 20)$, $(v_{11}, w_{11}) = (65, 18)$, $(v_{12}, w_{12}) = (95, 28)$, $(v_{13}, w_{13}) = (45, 8)$, $(v_{14}, w_{14}) = (85, 22)$, $(v_{15}, w_{15}) = (70, 25)$, $(v_{16}, w_{16}) = (110, 40)$, $(v_{17}, w_{17}) = (50, 14)$, $(v_{18}, w_{18}) = (60, 16)$, $(v_{19}, w_{19}) = (120, 50)$, $(v_{20}, w_{20}) = (100, 30)$

---

**Summary:**

Maximize total value of products allocated to shelves, subject to shelf capacity constraints, with integer allocation variables for each shelf-product pair, using the explicit data above.