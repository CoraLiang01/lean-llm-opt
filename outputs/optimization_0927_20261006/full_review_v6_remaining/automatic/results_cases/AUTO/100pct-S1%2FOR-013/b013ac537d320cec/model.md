Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName $j$ from products.csv) to be placed in storage area $i$ (StorageID $i$ from capacity.csv). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $S$ be the set of storage areas, with StorageID and Capacity as below (in source order):

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

- Let $P$ be the set of air conditioner types, with ProductName, Value, and Weight as below (in source order):

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
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
$$

**Objective:**

Maximize the total value of air conditioners allocated:
$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$
where $v_j$ is the Value of product $j$.

**Constraints:**

For each storage area $i \in S$ (with Capacity $c_i$):

$$
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in S
$$

where $w_j$ is the Weight (size) of product $j$.

**Variable Domains:**

$$
x_{ij} \in \{0, 1, 2, \ldots\} \qquad \forall i \in S, \forall j \in P
$$

**Explicit Data Used:**

- Storage areas (StorageID, Capacity):  
  1: 1083  
  2: 1840  
  3: 770  
  4: 1299  
  5: 1259  
  6: 543  
  7: 1831  
  8: 855  
  9: 619  
  10: 637  
  11: 935  
  12: 626  
  13: 1457  
  14: 1198  
  15: 837  

- Air conditioner types (ProductName, Value, Weight):  
  Window Unit: 4811, 114  
  Portable Unit: 1130, 200  
  Split System: 1611, 106  
  Ductless System: 3368, 256  
  Central AC: 2135, 268  
  Hybrid AC: 1046, 185  
  Geothermal AC: 4030, 299  
  Smart AC: 3761, 131  
  Evaporative Cooler: 3523, 139  
  Package Unit: 1701, 105  

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j \in P} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in S \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S, \forall j \in P
\end{align*}
$$

where all sets and parameters are as listed above, and all indices and coefficients are as retrieved.