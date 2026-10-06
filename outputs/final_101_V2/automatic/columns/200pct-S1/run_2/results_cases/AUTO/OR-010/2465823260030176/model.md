Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $x_{ij} \geq 0$.

**Parameters:**

- Sections (from capacity.csv, in order):

  | SectionID |
  |-----------|
  | 1         |
  | 2         |
  | 3         |
  | 4         |
  | 5         |
  | 6         |
  | 7         |
  | 8         |

  Let $S$ be the set of SectionIDs: $S = \{1,2,3,4,5,6,7,8\}$

  Section capacities:

  $C_1 = 100$, $C_2 = 150$, $C_3 = 120$, $C_4 = 130$, $C_5 = 90$, $C_6 = 110$, $C_7 = 160$, $C_8 = 140$

- Products (from products.csv, in order):

  | ProductName | Value | Weight |
  |-------------|-------|--------|
  | 1           | 10    | 2      |
  | 2           | 15    | 3      |
  | 3           | 8     | 1      |
  | 4           | 12    | 2      |
  | 5           | 20    | 4      |
  | 6           | 25    | 5      |
  | 7           | 5     | 1      |
  | 8           | 30    | 6      |
  | 9           | 18    | 3      |
  | 10          | 22    | 4      |

  Let $P$ be the set of ProductNames: $P = \{1,2,3,4,5,6,7,8,9,10\}$

  For each product $j$:
  - $v_j$ = Value of product $j$
  - $w_j$ = Weight (space requirement) of product $j$

**Decision Variables:**

$x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i \in S$, $j \in P$

---

### Mathematical Model

**Objective:**

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

**Subject to:**

For each section $i \in S$ (using SectionID and Capacity from capacity.csv):

$$
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
$$

That is, for each section:

- Section 1: $\sum_{j \in P} w_j x_{1j} \leq 100$
- Section 2: $\sum_{j \in P} w_j x_{2j} \leq 150$
- Section 3: $\sum_{j \in P} w_j x_{3j} \leq 120$
- Section 4: $\sum_{j \in P} w_j x_{4j} \leq 130$
- Section 5: $\sum_{j \in P} w_j x_{5j} \leq 90$
- Section 6: $\sum_{j \in P} w_j x_{6j} \leq 110$
- Section 7: $\sum_{j \in P} w_j x_{7j} \leq 160$
- Section 8: $\sum_{j \in P} w_j x_{8j} \leq 140$

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
$$

---

**Parameter Table (from products.csv):**

| ProductName | Value ($v_j$) | Weight ($w_j$) |
|-------------|--------------|---------------|
| 1           | 10           | 2             |
| 2           | 15           | 3             |
| 3           | 8            | 1             |
| 4           | 12           | 2             |
| 5           | 20           | 4             |
| 6           | 25           | 5             |
| 7           | 5            | 1             |
| 8           | 30           | 6             |
| 9           | 18           | 3             |
| 10          | 22           | 4             |

**Section Capacities (from capacity.csv):**

| SectionID ($i$) | Capacity ($C_i$) |
|-----------------|-----------------|
| 1               | 100             |
| 2               | 150             |
| 3               | 120             |
| 4               | 130             |
| 5               | 90              |
| 6               | 110             |
| 7               | 160             |
| 8               | 140             |

---

**Summary:**

Maximize total revenue from all products allocated to all sections, subject to each section's display space limit, with integer nonnegative allocation variables for each product-section pair. All coefficients and identifiers are as retrieved and in original order.