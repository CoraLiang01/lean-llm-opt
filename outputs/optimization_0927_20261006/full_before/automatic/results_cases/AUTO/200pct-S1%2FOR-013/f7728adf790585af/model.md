Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName from products.csv) to be placed in storage area $i$ (StorageID from capacity.csv). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Storage areas (from capacity.csv): $i \in \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$
- Air conditioner types (from products.csv): $j \in \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$
- $C_i$: Capacity of storage area $i$ (from capacity.csv)
- $v_j$: Value of air conditioner type $j$ (from products.csv)
- $w_j$: Weight (size) of air conditioner type $j$ (from products.csv)

**Data:**

Storage Areas and Capacities (capacity.csv, in source order):

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

Air Conditioner Types, Values, and Weights (products.csv, in source order):

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

**Mathematical Model:**

Objective:
\[
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} v_j \cdot x_{ij}
\]

Subject to, for each storage area $i$ (using StorageID as $i$):

\[
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

Where:

- $C_i$ is the Capacity for StorageID $i$ as listed above.
- $v_j$ and $w_j$ are the Value and Weight for ProductName $j$ as listed above.
- $x_{ij}$ is the integer number of units of air conditioner type $j$ placed in storage area $i$.

All variables, parameters, and constraints use the exact identifiers and coefficients as retrieved.