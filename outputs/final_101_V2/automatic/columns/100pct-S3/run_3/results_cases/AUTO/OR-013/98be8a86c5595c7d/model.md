Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName from products.csv) to be placed in storage area $i$ (StorageID from capacity.csv). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas $i$ (from capacity.csv, StorageID):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15  
  with capacities $C_i$ as below.

- Air conditioner types $j$ (from products.csv, ProductName):  
  Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit  
  with values $v_j$ and weights $w_j$ as below.

| StorageID ($i$) | Capacity ($C_i$) |
|---|---|
| 1 | 1083 |
| 2 | 1840 |
| 3 | 770 |
| 4 | 1299 |
| 5 | 1259 |
| 6 | 543 |
| 7 | 1831 |
| 8 | 855 |
| 9 | 619 |
| 10 | 637 |
| 11 | 935 |
| 12 | 626 |
| 13 | 1457 |
| 14 | 1198 |
| 15 | 837 |

| ProductName ($j$) | Value ($v_j$) | Weight ($w_j$) |
|---|---|---|
| Window Unit | 4811 | 114 |
| Portable Unit | 1130 | 200 |
| Split System | 1611 | 106 |
| Ductless System | 3368 | 256 |
| Central AC | 2135 | 268 |
| Hybrid AC | 1046 | 185 |
| Geothermal AC | 4030 | 299 |
| Smart AC | 3761 | 131 |
| Evaporative Cooler | 3523 | 139 |
| Package Unit | 1701 | 105 |

---

**Mathematical Model**

**Decision variables:**  
$x_{ij} \in \mathbb{Z}_{\geq 0}$, for all storage areas $i$ and product types $j$.

**Objective:**  
Maximize the total value of air conditioners allocated:
$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} v_j \cdot x_{ij}
$$

**Subject to:**

For each storage area $i$ (StorageID):

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

Where $C_i$ is the Capacity for storage area $i$, and $w_j$ is the Weight (size) of product $j$.

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

**All identifiers and coefficients are as retrieved above.**