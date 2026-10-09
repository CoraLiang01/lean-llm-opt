Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName $j$) to be placed in storage area $i$ (StorageID $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $S$ = set of storage areas (indexed by $i$), with StorageID and Capacity as below.
- $P$ = set of air conditioner types (indexed by $j$), with ProductName, Value, and Weight as below.
- $c_i$ = Capacity of storage area $i$.
- $v_j$ = Value of air conditioner type $j$.
- $w_j$ = Weight (size) of air conditioner type $j$.

---

### Objective

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

### Constraints

For each storage area $i \in S$:
$$
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i
$$

For all $i \in S$, $j \in P$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

### Parameter Tables

#### Storage Areas (from capacity.csv, in source order):

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

#### Air Conditioner Types (from products.csv, in source order):

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

**Decision variables:**  
$x_{ij}$: integer, $\geq 0$, for all StorageID $i$ and ProductName $j$.

---

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j \in P} w_j \cdot x_{ij} \leq c_i, \quad \forall i \in S \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in S,\, j \in P
\end{align*}
$$

with all parameters as listed above.