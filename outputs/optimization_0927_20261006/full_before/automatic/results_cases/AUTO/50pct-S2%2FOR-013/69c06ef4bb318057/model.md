Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Storage areas (from capacity.csv):

| StorageID |
|-----------|
| 1         |
| 2         |
| 3         |
| 4         |
| 5         |
| 6         |
| 7         |
| 8         |
| 9         |
| 10        |
| 11        |
| 12        |
| 13        |
| 14        |
| 15        |

- Storage area capacities (Capacity):

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

- Air conditioner types (from products.csv):

| ProductName           |
|-----------------------|
| Window Unit           |
| Portable Unit         |
| Split System          |
| Ductless System       |
| Central AC            |
| Hybrid AC             |
| Geothermal AC         |
| Smart AC              |
| Evaporative Cooler    |
| Package Unit          |

- Product values (Value):

| ProductName         | Value |
|---------------------|-------|
| Window Unit         | 4811  |
| Portable Unit       | 1130  |
| Split System        | 1611  |
| Ductless System     | 3368  |
| Central AC          | 2135  |
| Hybrid AC           | 1046  |
| Geothermal AC       | 4030  |
| Smart AC            | 3761  |
| Evaporative Cooler  | 3523  |
| Package Unit        | 1701  |

- Product sizes (Weight):

| ProductName         | Weight |
|---------------------|--------|
| Window Unit         | 114    |
| Portable Unit       | 200    |
| Split System        | 106    |
| Ductless System     | 256    |
| Central AC          | 268    |
| Hybrid AC           | 185    |
| Geothermal AC       | 299    |
| Smart AC            | 131    |
| Evaporative Cooler  | 139    |
| Package Unit        | 105    |

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} = \text{number of units of air conditioner type } j \text{ placed in storage area } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

where $i \in \{1,2,\ldots,15\}$ (StorageID), $j$ indexes the 10 ProductName values above.

---

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} v_j \cdot x_{ij}
$$

where $v_j$ is the Value for product $j$ as given above.

---

**Constraints:**

For each storage area $i$ (StorageID):

$$
\sum_{j} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,\ldots,15\}
$$

where $w_j$ is the Weight for product $j$ and $c_i$ is the Capacity for storage area $i$.

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**All parameters and identifiers are as retrieved above.**