Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName $j$) to be placed in storage area $i$ (StorageID $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Storage areas (from capacity.csv, in source order):

  | StorageID |
  |-----------|
  | 1         |
  | 2         |
  | 3         |
  | 4         |
  | 5         |
  | 6         |
  | 7         |
  | 8         |
  | 9         |
  | 10        |
  | 11        |
  | 12        |
  | 13        |
  | 14        |
  | 15        |

  Capacities $C_i$:

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

- Air conditioner types (from products.csv, in source order):

  | ProductName           | Value | Weight |
  |----------------------|-------|--------|
  | Window Unit          | 4811  | 114    |
  | Portable Unit        | 1130  | 200    |
  | Split System         | 1611  | 106    |
  | Ductless System      | 3368  | 256    |
  | Central AC           | 2135  | 268    |
  | Hybrid AC            | 1046  | 185    |
  | Geothermal AC        | 4030  | 299    |
  | Smart AC             | 3761  | 131    |
  | Evaporative Cooler   | 3523  | 139    |
  | Package Unit         | 1701  | 105    |

Let $S$ be the set of StorageIDs (as above), and $P$ be the set of ProductNames (as above).

Let $v_j$ be the Value of product $j$, and $w_j$ be the Weight of product $j$.

Let $C_i$ be the Capacity of storage area $i$.

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
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

**Parameter Table (source order):**

- StorageIDs and Capacities:

  1: 1083, 2: 1840, 3: 770, 4: 1299, 5: 1259, 6: 543, 7: 1831, 8: 855, 9: 619, 10: 637, 11: 935, 12: 626, 13: 1457, 14: 1198, 15: 837

- ProductNames, Values, Weights:

  - Window Unit: 4811, 114
  - Portable Unit: 1130, 200
  - Split System: 1611, 106
  - Ductless System: 3368, 256
  - Central AC: 2135, 268
  - Hybrid AC: 1046, 185
  - Geothermal AC: 4030, 299
  - Smart AC: 3761, 131
  - Evaporative Cooler: 3523, 139
  - Package Unit: 1701, 105

---

**Summary:**

- Maximize total value of air conditioners allocated to storage areas.
- For each storage area, total weight of allocated units cannot exceed its capacity.
- All allocations are nonnegative integers.