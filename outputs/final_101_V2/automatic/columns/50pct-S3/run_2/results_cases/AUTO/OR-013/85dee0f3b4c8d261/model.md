Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas $i$ (from capacity.csv, StorageID):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

- Air conditioner types $j$ (from products.csv, ProductName):  
  Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit

- Storage area capacities $C_i$ (from capacity.csv, Capacity):  
  $C_1 = 1083$, $C_2 = 1840$, $C_3 = 770$, $C_4 = 1299$, $C_5 = 1259$, $C_6 = 543$, $C_7 = 1831$, $C_8 = 855$, $C_9 = 619$, $C_{10} = 637$, $C_{11} = 935$, $C_{12} = 626$, $C_{13} = 1457$, $C_{14} = 1198$, $C_{15} = 837$

- Product values $v_j$ (from products.csv, Value):  
  Window Unit: 4811  
  Portable Unit: 1130  
  Split System: 1611  
  Ductless System: 3368  
  Central AC: 2135  
  Hybrid AC: 1046  
  Geothermal AC: 4030  
  Smart AC: 3761  
  Evaporative Cooler: 3523  
  Package Unit: 1701

- Product weights $w_j$ (from products.csv, Weight):  
  Window Unit: 114  
  Portable Unit: 200  
  Split System: 106  
  Ductless System: 256  
  Central AC: 268  
  Hybrid AC: 185  
  Geothermal AC: 299  
  Smart AC: 131  
  Evaporative Cooler: 139  
  Package Unit: 105

---

**Mathematical Model**

**Decision Variables:**  
$x_{ij} \in \mathbb{Z}_{\geq 0}$, for all storage areas $i$ and air conditioner types $j$

**Objective:**  
Maximize total value of air conditioners allocated:
$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} v_j \cdot x_{ij}
$$

**Subject to:**

For each storage area $i$ (StorageID):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,15\}
$$

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
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