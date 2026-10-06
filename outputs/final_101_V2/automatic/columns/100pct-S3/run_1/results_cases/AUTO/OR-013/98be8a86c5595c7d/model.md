Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName from products.csv) to be placed in storage area $i$ (StorageID from capacity.csv). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas $i \in \{1,2,\ldots,15\}$, with capacities $C_i$ from the Capacity column of capacity.csv.
- Air conditioner types $j \in \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$, with:
    - Value $v_j$ from the Value column of products.csv.
    - Weight $w_j$ from the Weight column of products.csv.

**Data:**

capacity.csv (in source order):

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

products.csv (in source order):

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
$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} v_j \cdot x_{ij}
$$

Subject to, for each storage area $i$ (StorageID):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

Where:

- $C_i$ is the Capacity for StorageID $i$ from capacity.csv.
- $v_j$ is the Value for ProductName $j$ from products.csv.
- $w_j$ is the Weight for ProductName $j$ from products.csv.

**Explicitly:**

Let $i$ index StorageID in the order: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15.

Let $j$ index ProductName in the order:
1. Window Unit
2. Portable Unit
3. Split System
4. Ductless System
5. Central AC
6. Hybrid AC
7. Geothermal AC
8. Smart AC
9. Evaporative Cooler
10. Package Unit

With the following coefficients:

| $j$ | ProductName         | $v_j$ | $w_j$ |
|-----|---------------------|-------|-------|
| 1   | Window Unit         | 4811  | 114   |
| 2   | Portable Unit       | 1130  | 200   |
| 3   | Split System        | 1611  | 106   |
| 4   | Ductless System     | 3368  | 256   |
| 5   | Central AC          | 2135  | 268   |
| 6   | Hybrid AC           | 1046  | 185   |
| 7   | Geothermal AC       | 4030  | 299   |
| 8   | Smart AC            | 3761  | 131   |
| 9   | Evaporative Cooler  | 3523  | 139   |
| 10  | Package Unit        | 1701  | 105   |

And for each $i$ (StorageID), $C_i$ as above.

**Decision variables:**
$$
x_{ij} = \text{number of units of air conditioner type } j \text{ placed in storage area } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

**Summary:**
- Maximize total value of air conditioners allocated.
- For each storage area, total weight of allocated units cannot exceed its capacity.
- All allocations are nonnegative integers.