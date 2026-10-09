Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are required to be nonnegative integers.

**Parameters:**

- Let $S$ be the set of storage areas (indexed by $i$), with StorageID and Capacity as below.
- Let $P$ be the set of air conditioner types (indexed by $j$), with ProductName, Value, and Weight as below.

**Storage Areas (from capacity.csv, in source order):**

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

**Air Conditioner Types (from products.csv, in source order):**

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

**Objective Function:**

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$.

**Constraints:**

For each storage area $i \in S$ (with Capacity $C_i$):

$$
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in S
$$

where $w_j$ is the Weight of product $j$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S, \forall j \in P
$$

---

#### Explicit Data Mapping

- $S = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$, with $C_i$ as above.
- $P = \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$, with $v_j$ and $w_j$ as above.

---

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{15} \sum_{j=1}^{10} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,15,\; j = 1,\ldots,10
\end{align*}
$$

where the mapping of $i$ and $j$ to StorageID and ProductName is as listed above, and $v_j$, $w_j$, $C_i$ are the corresponding Value, Weight, and Capacity.