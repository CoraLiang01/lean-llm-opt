Let $x_{ij}$ be the number of units of product $j$ (ProductName $j$) to be placed on shelf $i$ (ShelfID $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves (from capacity.csv, in source order):

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

- Products (from products.csv, in source order):

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

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of ProductName $j$.

**Constraints:**

For each shelf $i$ (ShelfID $i$):

$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the Weight of ProductName $j$, and $C_i$ is the Capacity of ShelfID $i$.

**Explicitly, using the retrieved data:**

For each $i \in \{1,2,3,4,5,6,7,8,9,10\}$:

$$
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
$$

with:

- $\text{Capacity}_1 = 500$
- $\text{Capacity}_2 = 700$
- $\text{Capacity}_3 = 600$
- $\text{Capacity}_4 = 800$
- $\text{Capacity}_5 = 550$
- $\text{Capacity}_6 = 900$
- $\text{Capacity}_7 = 650$
- $\text{Capacity}_8 = 750$
- $\text{Capacity}_9 = 820$
- $\text{Capacity}_{10} = 570$

and

- $\text{Weight}_1 = 10$, $\text{Weight}_2 = 20$, $\text{Weight}_3 = 5$, $\text{Weight}_4 = 15$, $\text{Weight}_5 = 25$, $\text{Weight}_6 = 30$, $\text{Weight}_7 = 12$, $\text{Weight}_8 = 35$, $\text{Weight}_9 = 10$, $\text{Weight}_{10} = 20$, $\text{Weight}_{11} = 18$, $\text{Weight}_{12} = 28$, $\text{Weight}_{13} = 8$, $\text{Weight}_{14} = 22$, $\text{Weight}_{15} = 25$, $\text{Weight}_{16} = 40$, $\text{Weight}_{17} = 14$, $\text{Weight}_{18} = 16$, $\text{Weight}_{19} = 50$, $\text{Weight}_{20} = 30$

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

---

**Summary of retrieved data:**

- Shelves: ShelfID 1–10, with respective capacities as above.
- Products: ProductName 1–20, with respective Value and Weight as above.
- Decision variables: $x_{ij}$ = number of units of product $j$ on shelf $i$, integer and nonnegative.
- Objective: maximize total value placed on all shelves.
- Constraints: total weight on each shelf does not exceed its capacity.