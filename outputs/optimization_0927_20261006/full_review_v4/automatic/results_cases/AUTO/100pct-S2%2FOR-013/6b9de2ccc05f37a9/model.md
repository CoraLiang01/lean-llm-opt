Let $x_{ij}$ be the number of units of air conditioner type $j$ (with ProductName as in products.csv) to be placed in storage area $i$ (with StorageID as in capacity.csv). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $S$ be the set of storage areas, with StorageID:
  $$
  S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15\}
  $$
- Let $P$ be the set of air conditioner types, with ProductName:
  $$
  P = \{\text{Window Unit},\ \text{Portable Unit},\ \text{Split System},\ \text{Ductless System},\ \text{Central AC},\ \text{Hybrid AC},\ \text{Geothermal AC},\ \text{Smart AC},\ \text{Evaporative Cooler},\ \text{Package Unit}\}
  $$
- For each storage area $i \in S$, let $C_i$ be its capacity:

  | StorageID | Capacity |
  |-----------|----------|
  | 1         | 1083     |
  | 2         | 1840     |
  | 3         | 770      |
  | 4         | 1299     |
  | 5         | 1259     |
  | 6         | 543      |
  | 7         | 1831     |
  | 8         | 855      |
  | 9         | 619      |
  | 10        | 637      |
  | 11        | 935      |
  | 12        | 626      |
  | 13        | 1457     |
  | 14        | 1198     |
  | 15        | 837      |

- For each product $j \in P$, let $v_j$ be its value and $w_j$ its weight (size):

  | ProductName           | Value | Weight |
  |-----------------------|-------|--------|
  | Window Unit           | 4811  | 114    |
  | Portable Unit         | 1130  | 200    |
  | Split System          | 1611  | 106    |
  | Ductless System       | 3368  | 256    |
  | Central AC            | 2135  | 268    |
  | Hybrid AC             | 1046  | 185    |
  | Geothermal AC         | 4030  | 299    |
  | Smart AC              | 3761  | 131    |
  | Evaporative Cooler    | 3523  | 139    |
  | Package Unit          | 1701  | 105    |

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
$$

**Objective:**

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

**Subject to:**

For each storage area $i \in S$:
$$
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
$$

For all $i \in S$, $j \in P$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Where:**

- $x_{ij}$ = number of units of product $j$ placed in storage area $i$
- $v_j$ = value of product $j$ (see table above)
- $w_j$ = weight (size) of product $j$ (see table above)
- $C_i$ = capacity of storage area $i$ (see table above)

All identifiers and coefficients are as retrieved and preserved in source order.