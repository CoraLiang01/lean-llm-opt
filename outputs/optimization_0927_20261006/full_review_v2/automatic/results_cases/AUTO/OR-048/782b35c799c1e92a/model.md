Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

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

---

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,15\},\ j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}
$$

**Objective:**

$$
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$ as given above.

**Constraints:**

For each storage area $i$ (StorageID):

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,15
$$

where $w_j$ is the Weight of product $j$ and $C_i$ is the Capacity of storage area $i$ as given above.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**All data used:**

- Storage areas (StorageID, Capacity):  
  1, 1083  
  2, 1840  
  3, 770  
  4, 1299  
  5, 1259  
  6, 543  
  7, 1831  
  8, 855  
  9, 619  
  10, 637  
  11, 935  
  12, 626  
  13, 1457  
  14, 1198  
  15, 837  

- Air conditioner types (ProductName, Value, Weight):  
  Window Unit, 4811, 114  
  Portable Unit, 1130, 200  
  Split System, 1611, 106  
  Ductless System, 3368, 256  
  Central AC, 2135, 268  
  Hybrid AC, 1046, 185  
  Geothermal AC, 4030, 299  
  Smart AC, 3761, 131  
  Evaporative Cooler, 3523, 139  
  Package Unit, 1701, 105  

---

**Summary:**

Maximize total value of air conditioners allocated to storage areas, subject to each area's capacity, using integer variables $x_{ij}$ for each (storage area, product) pair, with all coefficients and identifiers as above.