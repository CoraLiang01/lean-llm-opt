Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas (from capacity.csv):

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

**Decision Variables:**

$x_{ij} \in \mathbb{Z}_{\geq 0}$, for all storage areas $i \in \{1,\ldots,15\}$ and all air conditioner types $j \in \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$.

---

### Objective Function

$$
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of air conditioner type $j$.

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

### Constraints

For each storage area $i$ (StorageID):

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i
$$

where $w_j$ is the Weight of air conditioner type $j$, and $C_i$ is the Capacity of storage area $i$.

Explicitly, for each $i=1,\ldots,15$:

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
\leq C_i
$$

with $C_i$ as follows:

- $C_1 = 1083$
- $C_2 = 1840$
- $C_3 = 770$
- $C_4 = 1299$
- $C_5 = 1259$
- $C_6 = 543$
- $C_7 = 1831$
- $C_8 = 855$
- $C_9 = 619$
- $C_{10} = 637$
- $C_{11} = 935$
- $C_{12} = 626$
- $C_{13} = 1457$
- $C_{14} = 1198$
- $C_{15} = 837$

---

### Variable Domains

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,15\},\ j \in \{1,\ldots,10\}
$$

---

### Summary

Maximize the total value of air conditioners allocated to storage areas, subject to each area's capacity, using integer variables $x_{ij}$ for each combination of storage area and air conditioner type, with all coefficients and identifiers as above.