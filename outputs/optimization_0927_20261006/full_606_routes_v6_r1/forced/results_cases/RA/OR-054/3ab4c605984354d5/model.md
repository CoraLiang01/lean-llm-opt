Let $x_{ij}$ be the number of units of product $j$ placed on shelf $i$, where $i \in \{1,2,\ldots,10\}$ (ShelfID from capacity.csv) and $j \in \{1,2,\ldots,20\}$ (ProductName from products.csv).

**Parameters:**

- Shelf capacities (from capacity.csv):

  - Shelf 1: $C_1 = 750$
  - Shelf 2: $C_2 = 820$
  - Shelf 3: $C_3 = 570$
  - Shelf 4: $C_4 = 800$
  - Shelf 5: $C_5 = 550$
  - Shelf 6: $C_6 = 900$
  - Shelf 7: $C_7 = 650$
  - Shelf 8: $C_8 = 800$
  - Shelf 9: $C_9 = 850$
  - Shelf 10: $C_{10} = 900$

- Product values and weights (from products.csv):

  | ProductName ($j$) | Value ($v_j$) | Weight ($w_j$) |
  |-------------------|--------------|---------------|
  | 1                 | 55           | 10            |
  | 2                 | 75           | 20            |
  | 3                 | 65           | 5             |
  | 4                 | 60           | 15            |
  | 5                 | 80           | 25            |
  | 6                 | 90           | 35            |
  | 7                 | 40           | 45            |
  | 8                 | 100          | 55            |
  | 9                 | 55           | 65            |
  | 10                | 75           | 20            |
  | 11                | 110          | 18            |
  | 12                | 50           | 28            |
  | 13                | 60           | 8             |
  | 14                | 120          | 28            |
  | 15                | 70           | 25            |
  | 16                | 110          | 40            |
  | 17                | 50           | 55            |
  | 18                | 60           | 70            |
  | 19                | 120          | 85            |
  | 20                | 100          | 100           |

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

**Subject to:**

- Shelf capacity constraints (for each shelf $i$):
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

- Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\; j \in \{1,2,\ldots,20\}
\]

---

**Where:**

- $x_{ij}$ = number of units of product $j$ placed on shelf $i$
- $v_j$ = value of product $j$ (see table above)
- $w_j$ = weight of product $j$ (see table above)
- $C_i$ = capacity of shelf $i$ (see list above)

All variables, parameters, and constraints use the exact identifiers and coefficients as retrieved.