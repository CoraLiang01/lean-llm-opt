Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas (from capacity.csv):

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

- Capacities (from capacity.csv):

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

- Air conditioner types (from products.csv):

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

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} = \text{number of units of air conditioner type } j \text{ placed in storage area } i
$$

where $i \in \{\text{1, 2, ..., 15}\}$ (StorageID), $j \in \{\text{Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit}\}$ (ProductName).

All $x_{ij} \in \mathbb{Z}_{\geq 0}$.

---

**Objective Function:**

$$
\max \sum_{i \in \{\text{1,...,15}\}} \sum_{j \in \{\text{all ProductNames}\}} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$.

Explicitly:

$$
\max \sum_{i=1}^{15} \Big(
4811\, x_{i,\text{Window Unit}} +
1130\, x_{i,\text{Portable Unit}} +
1611\, x_{i,\text{Split System}} +
3368\, x_{i,\text{Ductless System}} +
2135\, x_{i,\text{Central AC}} +
1046\, x_{i,\text{Hybrid AC}} +
4030\, x_{i,\text{Geothermal AC}} +
3761\, x_{i,\text{Smart AC}} +
3523\, x_{i,\text{Evaporative Cooler}} +
1701\, x_{i,\text{Package Unit}}
\Big)
$$

---

**Constraints:**

For each storage area $i$ (StorageID):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i
$$

where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of storage area $i$.

Explicitly, for each $i = 1, ..., 15$:

$$
114\, x_{i,\text{Window Unit}} +
200\, x_{i,\text{Portable Unit}} +
106\, x_{i,\text{Split System}} +
256\, x_{i,\text{Ductless System}} +
268\, x_{i,\text{Central AC}} +
185\, x_{i,\text{Hybrid AC}} +
299\, x_{i,\text{Geothermal AC}} +
131\, x_{i,\text{Smart AC}} +
139\, x_{i,\text{Evaporative Cooler}} +
105\, x_{i,\text{Package Unit}}
\leq \text{Capacity}_i
$$

where $\text{Capacity}_i$ is as listed above for each StorageID.

---

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i, j
$$

---

**All data used:**

- Storage areas and capacities (from capacity.csv, in source order):

    1: 1083, 2: 1840, 3: 770, 4: 1299, 5: 1259, 6: 543, 7: 1831, 8: 855, 9: 619, 10: 637, 11: 935, 12: 626, 13: 1457, 14: 1198, 15: 837

- Air conditioner types, values, and weights (from products.csv, in source order):

    - Window Unit: Value 4811, Weight 114
    - Portable Unit: Value 1130, Weight 200
    - Split System: Value 1611, Weight 106
    - Ductless System: Value 3368, Weight 256
    - Central AC: Value 2135, Weight 268
    - Hybrid AC: Value 1046, Weight 185
    - Geothermal AC: Value 4030, Weight 299
    - Smart AC: Value 3761, Weight 131
    - Evaporative Cooler: Value 3523, Weight 139
    - Package Unit: Value 1701, Weight 105

---

**Summary:**

Maximize total value of air conditioners allocated to storage areas, subject to each area's capacity, with integer nonnegative allocation variables for each product-storage pair, using the exact identifiers and coefficients as retrieved.