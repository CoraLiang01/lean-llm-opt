Let $x_{ij}$ be the number of units of air conditioner type $j$ to be placed in storage area $i$. All $x_{ij}$ are integer and nonnegative.

**Sets and Indices:**
- $i \in \{\text{1}, \text{2}, \ldots, \text{15}\}$ (StorageID from capacity.csv)
- $j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}$ (ProductName from products.csv)

**Parameters:**
- $C_i$ = Capacity of storage area $i$ (from Capacity column)
- $v_j$ = Value of air conditioner type $j$ (from Value column)
- $w_j$ = Weight (size) of air conditioner type $j$ (from Weight column)

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
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{1},\ldots,\text{15}\},\ j \in \{\text{Window Unit}, \ldots, \text{Package Unit}\}
$$

**Objective:**
$$
\max \sum_{i \in \{\text{1},\ldots,\text{15}\}} \sum_{j \in \{\text{Window Unit}, \ldots, \text{Package Unit}\}} v_j \cdot x_{ij}
$$

**Subject to:**

For each storage area $i$ (using StorageID as $i$):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{1},\ldots,\text{15}\}
$$

Where:
- $C_i$ is the Capacity for StorageID $i$ (see table above)
- $w_j$ is the Weight for ProductName $j$ (see table above)
- $v_j$ is the Value for ProductName $j$ (see table above)

**Variable Domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**All identifiers, coefficients, and constraints are as retrieved and in source order.**