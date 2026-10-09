Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

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

  $$
  \begin{align*}
  C_1 &= 1083 \\
  C_2 &= 1840 \\
  C_3 &= 770 \\
  C_4 &= 1299 \\
  C_5 &= 1259 \\
  C_6 &= 543 \\
  C_7 &= 1831 \\
  C_8 &= 855 \\
  C_9 &= 619 \\
  C_{10} &= 637 \\
  C_{11} &= 935 \\
  C_{12} &= 626 \\
  C_{13} &= 1457 \\
  C_{14} &= 1198 \\
  C_{15} &= 837 \\
  \end{align*}
  $$

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

Let $J$ be the set of product types (in the above order), and $I$ the set of storage areas (in the above order).

**Mathematical Model:**

**Objective:**
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$
where $v_j$ is the Value of product $j$.

**Subject to:**

For each storage area $i \in I$ (using StorageID and Capacity from capacity.csv):

$$
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
$$

where $w_j$ is the Weight of product $j$ and $C_i$ is the Capacity of storage area $i$.

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

**Explicit Data:**

- Storage areas $I$ (in source order): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
- Capacities $C_i$ (in source order): 1083, 1840, 770, 1299, 1259, 543, 1831, 855, 619, 637, 935, 626, 1457, 1198, 837
- Product types $J$ (in source order): Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit
- Values $v_j$ (in source order): 4811, 1130, 1611, 3368, 2135, 1046, 4030, 3761, 3523, 1701
- Weights $w_j$ (in source order): 114, 200, 106, 256, 268, 185, 299, 131, 139, 105

**Summary:**

Maximize total value of air conditioners allocated to storage areas, subject to each area's capacity, with integer nonnegative allocation variables for each product-storage pair.