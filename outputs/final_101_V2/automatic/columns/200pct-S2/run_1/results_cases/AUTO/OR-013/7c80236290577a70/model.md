Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName from products.csv) to be placed in storage area $i$ (StorageID from capacity.csv). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Let $S$ be the set of storage areas, indexed by $i$ (StorageID: 1, 2, ..., 15).
- Let $P$ be the set of air conditioner types, indexed by $j$ (ProductName: Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit).
- $c_i$ = Capacity of storage area $i$ (from capacity.csv).
- $v_j$ = Value of air conditioner type $j$ (from products.csv).
- $w_j$ = Weight (size) of air conditioner type $j$ (from products.csv).

**Data:**

Storage Areas (from capacity.csv):

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

Air Conditioner Types (from products.csv):

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

**Mathematical Model:**

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

Subject to, for each storage area $i$:
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in S
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
\]

**Where:**

- $x_{ij}$: Number of units of air conditioner type $j$ placed in storage area $i$ (integer, $\geq 0$)
- $v_j$: Value of air conditioner type $j$ (see table above)
- $w_j$: Weight (size) of air conditioner type $j$ (see table above)
- $c_i$: Capacity of storage area $i$ (see table above)

**All identifiers and coefficients are as retrieved.**