Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $S$ = set of storage areas (indexed by $i$), with StorageID from capacity.csv
- $P$ = set of air conditioner types (indexed by $j$), with ProductName from products.csv
- $c_i$ = Capacity of storage area $i$ (from capacity.csv, column "Capacity")
- $v_j$ = Value of air conditioner type $j$ (from products.csv, column "Value")
- $w_j$ = Weight (size) of air conditioner type $j$ (from products.csv, column "Weight")

**Sets and Data:**

Storage Areas (from capacity.csv, in source order):

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

Capacities:

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
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
$$

**Objective:**

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

**Subject to:**

For each storage area $i \in S$ (using StorageID and Capacity from capacity.csv):

$$
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in S
$$

**Variable Domains:**

$$
x_{ij} \in \{0, 1, 2, \ldots\} \qquad \forall i \in S, \forall j \in P
$$

---

**Where:**

- $S = \{1, 2, \ldots, 15\}$ (StorageID from capacity.csv, in source order)
- $P =$ {Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit} (ProductName from products.csv, in source order)
- $c_i$ = Capacity of storage area $i$ (see table above)
- $v_j$ = Value of air conditioner type $j$ (see table above)
- $w_j$ = Weight of air conditioner type $j$ (see table above)

All data and identifiers are preserved in original file and row order.