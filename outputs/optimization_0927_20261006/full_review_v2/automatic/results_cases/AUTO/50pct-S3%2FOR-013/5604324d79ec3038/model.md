Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- $S$ = set of storage areas (StorageID):  
  $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15\}$
- $P$ = set of air conditioner types (ProductName):  
  $P = \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$
- $c_i$ = capacity of storage area $i$ (from Capacity column)
- $v_j$ = value of air conditioner type $j$ (from Value column)
- $w_j$ = size (Weight) of air conditioner type $j$

**Data:**

Storage Areas (from capacity.csv, in source order):

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

Air Conditioner Types (from products.csv, in source order):

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

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \; j \in P
$$

**Objective:**

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

**Subject to:**

For each storage area $i \in S$:
$$
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i
$$

For all $i \in S$, $j \in P$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Explicitly, with all identifiers and coefficients:**

Let $x_{ij}$ = number of units of ProductName $j$ in StorageID $i$.

**Objective:**
$$
\max \Bigg[
\sum_{i=1}^{15} \Big(
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
\Bigg]
$$

**For each StorageID $i$ (in source order):**

- For StorageID 1 (Capacity 1083):
  $$
  114\, x_{1,\text{Window Unit}} +
  200\, x_{1,\text{Portable Unit}} +
  106\, x_{1,\text{Split System}} +
  256\, x_{1,\text{Ductless System}} +
  268\, x_{1,\text{Central AC}} +
  185\, x_{1,\text{Hybrid AC}} +
  299\, x_{1,\text{Geothermal AC}} +
  131\, x_{1,\text{Smart AC}} +
  139\, x_{1,\text{Evaporative Cooler}} +
  105\, x_{1,\text{Package Unit}}
  \leq 1083
  $$
- For StorageID 2 (Capacity 1840):
  $$
  114\, x_{2,\text{Window Unit}} + \ldots + 105\, x_{2,\text{Package Unit}} \leq 1840
  $$
- For StorageID 3 (Capacity 770):
  $$
  114\, x_{3,\text{Window Unit}} + \ldots + 105\, x_{3,\text{Package Unit}} \leq 770
  $$
- For StorageID 4 (Capacity 1299):
  $$
  114\, x_{4,\text{Window Unit}} + \ldots + 105\, x_{4,\text{Package Unit}} \leq 1299
  $$
- For StorageID 5 (Capacity 1259):
  $$
  114\, x_{5,\text{Window Unit}} + \ldots + 105\, x_{5,\text{Package Unit}} \leq 1259
  $$
- For StorageID 6 (Capacity 543):
  $$
  114\, x_{6,\text{Window Unit}} + \ldots + 105\, x_{6,\text{Package Unit}} \leq 543
  $$
- For StorageID 7 (Capacity 1831):
  $$
  114\, x_{7,\text{Window Unit}} + \ldots + 105\, x_{7,\text{Package Unit}} \leq 1831
  $$
- For StorageID 8 (Capacity 855):
  $$
  114\, x_{8,\text{Window Unit}} + \ldots + 105\, x_{8,\text{Package Unit}} \leq 855
  $$
- For StorageID 9 (Capacity 619):
  $$
  114\, x_{9,\text{Window Unit}} + \ldots + 105\, x_{9,\text{Package Unit}} \leq 619
  $$
- For StorageID 10 (Capacity 637):
  $$
  114\, x_{10,\text{Window Unit}} + \ldots + 105\, x_{10,\text{Package Unit}} \leq 637
  $$
- For StorageID 11 (Capacity 935):
  $$
  114\, x_{11,\text{Window Unit}} + \ldots + 105\, x_{11,\text{Package Unit}} \leq 935
  $$
- For StorageID 12 (Capacity 626):
  $$
  114\, x_{12,\text{Window Unit}} + \ldots + 105\, x_{12,\text{Package Unit}} \leq 626
  $$
- For StorageID 13 (Capacity 1457):
  $$
  114\, x_{13,\text{Window Unit}} + \ldots + 105\, x_{13,\text{Package Unit}} \leq 1457
  $$
- For StorageID 14 (Capacity 1198):
  $$
  114\, x_{14,\text{Window Unit}} + \ldots + 105\, x_{14,\text{Package Unit}} \leq 1198
  $$
- For StorageID 15 (Capacity 837):
  $$
  114\, x_{15,\text{Window Unit}} + \ldots + 105\, x_{15,\text{Package Unit}} \leq 837
  $$

**Variable domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,15\},\; j \in P
$$

All coefficients and identifiers are as retrieved and in source order.