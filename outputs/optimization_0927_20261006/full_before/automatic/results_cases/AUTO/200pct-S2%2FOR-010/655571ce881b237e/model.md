Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Let $S$ be the set of sections (indexed by SectionID), and $P$ the set of products (indexed by ProductName).

Let $c_i$ be the capacity of section $i$ (from Capacity in capacity.csv).

Let $v_j$ be the value (price) of product $j$ (from Value in products.csv).

Let $w_j$ be the shelf space requirement of product $j$ (from Weight in products.csv).

---

**Sets:**

- $S = \{1, 2, 3, 4, 5, 6, 7, 8\}$ (SectionID from capacity.csv)
- $P = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$ (ProductName from products.csv)

**Parameters:**

- Section capacities:
  - $c_1 = 100$
  - $c_2 = 150$
  - $c_3 = 120$
  - $c_4 = 130$
  - $c_5 = 90$
  - $c_6 = 110$
  - $c_7 = 160$
  - $c_8 = 140$

- Product values and shelf space requirements:

| ProductName ($j$) | $v_j$ (Value) | $w_j$ (Weight) |
|:-----------------:|:-------------:|:--------------:|
| 1                 | 10            | 2              |
| 2                 | 15            | 3              |
| 3                 | 8             | 1              |
| 4                 | 12            | 2              |
| 5                 | 20            | 4              |
| 6                 | 25            | 5              |
| 7                 | 5             | 1              |
| 8                 | 30            | 6              |
| 9                 | 18            | 3              |
| 10                | 22            | 4              |

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

**Subject to:**

- Section capacity constraints (for each section $i$):
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in S
\]

- Non-negativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
\]

---

**Where:**

- $x_{ij}$: Number of units of product $j$ to be placed in section $i$ (integer, $\geq 0$)
- $v_j$: Value of product $j$ (see table above)
- $w_j$: Shelf space requirement of product $j$ (see table above)
- $c_i$: Capacity of section $i$ (see list above)

All identifiers and coefficients are as retrieved from the source files, in original order.