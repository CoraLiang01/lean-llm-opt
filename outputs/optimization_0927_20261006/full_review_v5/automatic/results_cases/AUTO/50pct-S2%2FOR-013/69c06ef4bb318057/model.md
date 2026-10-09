Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName from products.csv) to be placed in storage area $i$ (StorageID from capacity.csv). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $S$ = set of storage areas (StorageID):  
  $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15\}$
- $P$ = set of air conditioner types (ProductName):  
  $P = \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}$
- $v_j$ = value of one unit of product $j$ (from products.csv)
- $w_j$ = size (Weight) of one unit of product $j$ (from products.csv)
- $C_i$ = capacity of storage area $i$ (from capacity.csv)

**Data:**

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

| ProductName           | Value | Weight |
|---------------------- |-------|--------|
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

**Mathematical Model:**

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
$$

**Objective:**
$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

**Constraints:**

For each storage area $i \in S$:
$$
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
$$

**Variable Domains:**
$$
x_{ij} \in \{0, 1, 2, \ldots\} \quad \forall i \in S, \forall j \in P
$$

**Where:**

- $v_j$ and $w_j$ are as given in the table above for each product $j$.
- $C_i$ is as given in the table above for each storage area $i$.

This model allocates air conditioner units to storage areas to maximize total value, subject to each area's capacity.