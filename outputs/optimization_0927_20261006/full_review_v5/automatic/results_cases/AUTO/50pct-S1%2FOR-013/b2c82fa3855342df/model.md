#### Sets and Indices

- Let $I$ be the set of storage areas, indexed by $i$, with StorageID as below.
- Let $J$ be the set of air conditioner types, indexed by $j$, with ProductName as below.

#### Parameters

- $c_i$: Capacity of storage area $i$ (from Capacity column, by StorageID)
- $v_j$: Value of one unit of air conditioner type $j$ (from Value column, by ProductName)
- $w_j$: Size (Weight) of one unit of air conditioner type $j$ (from Weight column, by ProductName)

#### Decision Variables

- $x_{ij}$: Number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

---

### Objective

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

---

### Constraints

#### 1. Storage Area Capacity Constraints

For each storage area $i$:
$$
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
$$

#### 2. Nonnegativity and Integrality

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

---

### Parameter Tables

#### Storage Areas (from capacity.csv, source order):

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

#### Air Conditioner Types (from products.csv, source order):

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

$$
\begin{align*}
\max \quad & \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\end{align*}
$$

Where:

- $I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$
- $J =$ {Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit}
- $c_i$, $v_j$, $w_j$ as in the tables above.