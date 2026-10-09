#### Sets and Indices

- Let $I$ be the set of storage areas, indexed by $i$, with StorageID as below.
- Let $J$ be the set of air conditioner types, indexed by $j$, with ProductName as below.

#### Parameters

- $c_i$ = Capacity of storage area $i$ (from Capacity column, by StorageID)
- $v_j$ = Value of air conditioner type $j$ (from Value column, by ProductName)
- $w_j$ = Weight (size) of air conditioner type $j$ (from Weight column, by ProductName)

#### Decision Variables

- $x_{ij}$ = Number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

---

### Objective

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

---

### Constraints

#### 1. Storage Area Capacity Constraints

For each storage area $i \in I$:
$$
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i
$$

#### 2. Integrality and Nonnegativity

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
$$

---

### Parameter Tables (in source order)

#### Storage Areas (capacity.csv)

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

#### Air Conditioner Types (products.csv)

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

### Complete Model

Maximize:
$$
\sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

Subject to, for all $i \in I$:
$$
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i
$$

and
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
$$

Where:

- $I = \{$1, 2, 3, ..., 15$\}$ (StorageID)
- $J = \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$
- $c_i$, $v_j$, $w_j$ as given in the tables above.