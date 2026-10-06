Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName $j$) to be placed in storage area $i$ (StorageID $i$). All $x_{ij}$ are integer and nonnegative.

Sets:
- $i \in \{1,2,\ldots,15\}$ (StorageID from capacity.csv)
- $j \in \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$ (ProductName from products.csv)

Parameters:
- $c_i$: Capacity of storage area $i$ (from capacity.csv)
- $v_j$: Value of air conditioner type $j$ (from products.csv)
- $w_j$: Weight (size) of air conditioner type $j$ (from products.csv)

Data:

Storage Areas (capacity.csv, in source order):

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

Air Conditioner Types (products.csv, in source order):

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

Mathematical Model:

Objective:
\[
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j \cdot x_{ij}
\]
where $v_j$ is as above for each ProductName.

Subject to (for each storage area $i$):
\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,15\}
\]
where $w_j$ is as above for each ProductName, and $c_i$ is as above for each StorageID.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
\]

All data and constraints are included as required.