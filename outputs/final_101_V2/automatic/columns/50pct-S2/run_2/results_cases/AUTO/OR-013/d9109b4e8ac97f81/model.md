Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

Parameters (from the data):

- Storage areas $i$ (StorageID): 1, 2, 3, ..., 15
- Air conditioner types $j$ (ProductName): 
  - Window Unit
  - Portable Unit
  - Split System
  - Ductless System
  - Central AC
  - Hybrid AC
  - Geothermal AC
  - Smart AC
  - Evaporative Cooler
  - Package Unit
- Value $v_j$ and Weight $w_j$ for each product $j$:

| ProductName           | Value | Weight |
|----------------------|-------|--------|
| Window Unit          | 4811  | 114    |
| Portable Unit        | 1130  | 200    |
| Split System         | 1611  | 106    |
| Ductless System      | 3368  | 256    |
| Central AC           | 2135  | 268    |
| Hybrid AC            | 1046  | 185    |
| Geothermal AC        | 4030  | 299    |
| Smart AC             | 3761  | 131    |
| Evaporative Cooler   | 3523  | 139    |
| Package Unit         | 1701  | 105    |

- Storage area capacities $C_i$:

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

Model:

Objective:
\[
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j x_{ij}
\]

Subject to, for each storage area $i$:
\[
\sum_{j=1}^{10} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,15\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:
- $v_j$ and $w_j$ are as given above for each ProductName $j$.
- $C_i$ is the Capacity for each StorageID $i$.

All data and indices are as retrieved and preserved from the original files.