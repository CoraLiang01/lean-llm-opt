Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas (from capacity.csv, in order):

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

  Capacities $C_i$:

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

Let $V_j$ be the Value and $W_j$ the Weight for product $j$.

---

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,15\},\ j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}
$$

**Objective:**

$$
\max \sum_{i=1}^{15} \sum_{j=1}^{10} V_j \cdot x_{ij}
$$

where $V_j$ is as above for each product.

**Constraints:**

For each storage area $i$ (StorageID):

$$
\sum_{j=1}^{10} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

where $W_j$ is as above for each product, and $C_i$ is the Capacity for each StorageID.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Parameter Table (for reference):**

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