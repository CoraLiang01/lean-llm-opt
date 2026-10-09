Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Storage areas (from capacity.csv, in order):

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

- Air conditioner types (from products.csv, in order):

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

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{\text{StorageID}\},\ j \in \{\text{ProductName}\}
$$

**Objective Function:**

$$
\max \sum_{i \in \{\text{StorageID}\}} \sum_{j \in \{\text{ProductName}\}} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$.

**Constraints:**

For each storage area $i$ (StorageID):

$$
\sum_{j \in \{\text{ProductName}\}} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{\text{StorageID}\}
$$

where $w_j$ is the Weight of product $j$, and $c_i$ is the Capacity of storage area $i$.

**Explicitly, using the retrieved data:**

Let $i$ index StorageID $\in \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$

Let $j$ index ProductName $\in \{$

- Window Unit,
- Portable Unit,
- Split System,
- Ductless System,
- Central AC,
- Hybrid AC,
- Geothermal AC,
- Smart AC,
- Evaporative Cooler,
- Package Unit
$\}$

Let $v_j$ and $w_j$ be as in the table above, and $c_i$ as in the storage table.

**Model:**

$$
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

subject to

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i \qquad \forall i = 1,\ldots,15
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,15;\ j = 1,\ldots,10
$$

**Where:**

- $v_j$ and $w_j$ are as follows (in order):

  1. Window Unit: $v_1 = 4811$, $w_1 = 114$
  2. Portable Unit: $v_2 = 1130$, $w_2 = 200$
  3. Split System: $v_3 = 1611$, $w_3 = 106$
  4. Ductless System: $v_4 = 3368$, $w_4 = 256$
  5. Central AC: $v_5 = 2135$, $w_5 = 268$
  6. Hybrid AC: $v_6 = 1046$, $w_6 = 185$
  7. Geothermal AC: $v_7 = 4030$, $w_7 = 299$
  8. Smart AC: $v_8 = 3761$, $w_8 = 131$
  9. Evaporative Cooler: $v_9 = 3523$, $w_9 = 139$
  10. Package Unit: $v_{10} = 1701$, $w_{10} = 105$

- $c_i$ is the Capacity for StorageID $i$ as listed above.

**All variables $x_{ij}$ are nonnegative integers.**