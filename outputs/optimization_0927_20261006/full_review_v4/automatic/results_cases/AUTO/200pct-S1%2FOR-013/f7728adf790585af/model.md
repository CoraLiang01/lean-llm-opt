Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName $j$) to be placed in storage area $i$ (StorageID $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Storage areas (from capacity.csv, in source order):

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

  $C_1 = 1083$, $C_2 = 1840$, $C_3 = 770$, $C_4 = 1299$, $C_5 = 1259$, $C_6 = 543$, $C_7 = 1831$, $C_8 = 855$, $C_9 = 619$, $C_{10} = 637$, $C_{11} = 935$, $C_{12} = 626$, $C_{13} = 1457$, $C_{14} = 1198$, $C_{15} = 837$

- Air conditioner types (from products.csv, in source order):

  | ProductName           | Value | Weight |
  |---------------------- |-------|--------|
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

Let $V_j$ be the Value and $W_j$ the Weight for product $j$ as above.

---

**Mathematical Model:**

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} V_j \cdot x_{ij}
$$

**Subject to:**

For each storage area $i$ (StorageID as above):

$$
\sum_{j} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,15\}
$$

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

**Where:**

- $V_j$ and $W_j$ are as given above for each ProductName $j$.
- $C_i$ is the Capacity for StorageID $i$ as given above.
- $x_{ij}$ is the integer number of units of product $j$ in storage area $i$.

**All identifiers, coefficients, and constraints are as retrieved and in original order.**