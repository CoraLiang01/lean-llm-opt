Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName from products.csv) to be placed in storage area $i$ (StorageID from capacity.csv). All $x_{ij}$ are integer and $\geq 0$.

Indices:
- $i \in \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$ (StorageID from capacity.csv)
- $j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}$ (ProductName from products.csv)

Parameters:
- $c_i$ = Capacity of storage area $i$ (from capacity.csv)
- $v_j$ = Value of air conditioner type $j$ (from products.csv)
- $w_j$ = Weight (size) of air conditioner type $j$ (from products.csv)

Data:

capacity.csv

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

products.csv

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

Model:

Objective:
\[
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \ldots, \text{Package Unit}\}} v_j \cdot x_{ij}
\]

Subject to, for each storage area $i$:
\[
\sum_{j} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,15\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:

- $c_i$ is as given in the Capacity column for StorageID $i$.
- $v_j$ and $w_j$ are as given in the Value and Weight columns for ProductName $j$.

All data is used as retrieved, with no omitted rows or columns.