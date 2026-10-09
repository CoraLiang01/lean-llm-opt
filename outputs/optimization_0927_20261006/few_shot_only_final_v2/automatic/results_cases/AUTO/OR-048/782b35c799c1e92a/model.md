Let  
- $i$ index storage areas (StorageID from capacity.csv, $i \in \{1,2,\ldots,15\}$),  
- $j$ index air conditioner types (ProductName from products.csv, $j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}$),  
- $x_{ij}$ = number of units of air conditioner type $j$ placed in storage area $i$ (integer, $\geq 0$).

Parameters:  
- $c_i$ = capacity of storage area $i$ (from capacity.csv)  
- $v_j$ = value of air conditioner type $j$ (from products.csv)  
- $w_j$ = weight (size) of air conditioner type $j$ (from products.csv)

Data:

From capacity.csv:  
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

From products.csv:  
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

**Decision Variables:**  
$x_{ij} \in \mathbb{Z}_{\geq 0}$, for all storage areas $i$ and air conditioner types $j$

**Objective:**  
Maximize total value:
$$
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

**Subject to:**

For each storage area $i$ (for $i=1,\ldots,15$):
$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i
$$

For all $i=1,\ldots,15$, $j=1,\ldots,10$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

**Where:**

- $c_i$ is the capacity of storage area $i$ (see table above)
- $v_j$ is the value of air conditioner type $j$ (see table above)
- $w_j$ is the weight of air conditioner type $j$ (see table above)
- $x_{ij}$ is the number of units of air conditioner type $j$ placed in storage area $i$ (integer, $\geq 0$)