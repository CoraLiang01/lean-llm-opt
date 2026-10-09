Let:
- Storage areas be indexed by i ∈ {1, 2, ..., 15}, with StorageID and Capacity as given in capacity.csv.
- Air conditioner types be indexed by j ∈ {1, ..., 10}, with ProductName, Value, and Weight as given in products.csv.
- x_{ij} = number of units of air conditioner type j placed in storage area i (integer, x_{ij} ≥ 0).

Data (in source order):

Storage Areas (from capacity.csv):

| i  | StorageID | Capacity |
|----|-----------|----------|
| 1  | 1         | 1083     |
| 2  | 2         | 1840     |
| 3  | 3         | 770      |
| 4  | 4         | 1299     |
| 5  | 5         | 1259     |
| 6  | 6         | 543      |
| 7  | 7         | 1831     |
| 8  | 8         | 855      |
| 9  | 9         | 619      |
| 10 | 10        | 637      |
| 11 | 11        | 935      |
| 12 | 12        | 626      |
| 13 | 13        | 1457     |
| 14 | 14        | 1198     |
| 15 | 15        | 837      |

Products (from products.csv):

| j  | ProductName         | Value | Weight |
|----|---------------------|-------|--------|
| 1  | Window Unit         | 4811  | 114    |
| 2  | Portable Unit       | 1130  | 200    |
| 3  | Split System        | 1611  | 106    |
| 4  | Ductless System     | 3368  | 256    |
| 5  | Central AC          | 2135  | 268    |
| 6  | Hybrid AC           | 1046  | 185    |
| 7  | Geothermal AC       | 4030  | 299    |
| 8  | Smart AC            | 3761  | 131    |
| 9  | Evaporative Cooler  | 3523  | 139    |
| 10 | Package Unit        | 1701  | 105    |

Decision variables:
x_{ij} ∈ {0, 1, 2, ...} for all i ∈ {1,...,15}, j ∈ {1,...,10}

Objective:
Maximize total value of air conditioners allocated:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{15} \sum_{j=1}^{10} \text{Value}_j \cdot x_{ij}
\]
where Value_j is as given above for each product j.

Constraints:
For each storage area i (with StorageID and Capacity as above):
\[
\sum_{j=1}^{10} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
where Weight_j is as given above for each product j, and Capacity_i is as given above for each StorageID.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,...,15\},\ j \in \{1,...,10\}
\]

Explicitly, for each i (StorageID) and j (ProductName), use the corresponding Value and Weight from the tables above.

This is a pure integer programming model with variables x_{ij} representing the number of units of each air conditioner type j to be placed in storage area i, maximizing total value, subject to the capacity of each storage area.