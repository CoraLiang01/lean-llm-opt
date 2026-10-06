**Sets:**

- Let $I$ be the set of storage areas, indexed by StorageID from capacity.csv:  
  $I = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15\}$
- Let $J$ be the set of air conditioner types, indexed by ProductName from products.csv:  
  $J = \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}$

**Parameters:**

- $C_i$ = Capacity of storage area $i$ (from capacity.csv)
- $v_j$ = Value of air conditioner type $j$ (from products.csv)
- $w_j$ = Weight (size) of air conditioner type $j$ (from products.csv)

**Decision Variables:**

- $x_{ij}$ = Number of units of air conditioner type $j$ to be placed in storage area $i$  
  ($x_{ij} \in \mathbb{Z}_{\geq 0}$, integer and nonnegative)

---

### Mathematical Model

**Objective:**

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

**Subject to:**

For each storage area $i \in I$:
$$
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i
$$

For all $i \in I$, $j \in J$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

#### Data

**Storage Areas (capacity.csv):**

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

**Air Conditioner Types (products.csv):**

| ProductName         | Value | Weight |
|---------------------|-------|--------|
| Window Unit         | 4811  | 114    |
| Portable Unit       | 1130  | 200    |
| Split System        | 1611  | 106    |
| Ductless System     | 3368  | 256    |
| Central AC          | 2135  | 268    |
| Hybrid AC           | 1046  | 185    |
| Geothermal AC       | 4030  | 299    |
| Smart AC            | 3761  | 131    |
| Evaporative Cooler  | 3523  | 139    |
| Package Unit        | 1701  | 105    |

---

**Summary of Model:**

- Maximize total value of air conditioners allocated to all storage areas.
- For each storage area, the total size (weight) of all units placed cannot exceed its capacity.
- All decision variables are nonnegative integers.