Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas $i \in \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$, with capacities $C_i$:

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

- Air conditioner types $j$ (ProductName), with value $v_j$ and size $w_j$:

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

**Model:**

Maximize total value:
$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{all ProductNames}\}} v_j \cdot x_{ij}
$$

Subject to, for each storage area $i$:
$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

And integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

**Where:**

- $v_j$ and $w_j$ are as in the table above.
- $C_i$ is the Capacity for StorageID $i$ as in the table above.
- $x_{ij}$ is the number of units of air conditioner type $j$ to place in storage area $i$.

All variables and parameters use the exact identifiers and coefficients as retrieved.