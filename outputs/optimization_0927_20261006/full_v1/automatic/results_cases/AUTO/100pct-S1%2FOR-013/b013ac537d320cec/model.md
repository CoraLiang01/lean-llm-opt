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

  Capacities:

  $C_1 = 1083$, $C_2 = 1840$, $C_3 = 770$, $C_4 = 1299$, $C_5 = 1259$, $C_6 = 543$, $C_7 = 1831$, $C_8 = 855$, $C_9 = 619$, $C_{10} = 637$, $C_{11} = 935$, $C_{12} = 626$, $C_{13} = 1457$, $C_{14} = 1198$, $C_{15} = 837$

- Air conditioner types (from products.csv, in source order):

  | ProductName           | Value | Weight |
  |-----------------------|-------|--------|
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

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,15\},\ j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}
$$

**Objective:**

$$
\max \sum_{i=1}^{15} \Big( 4811\, x_{i,\text{Window Unit}} + 1130\, x_{i,\text{Portable Unit}} + 1611\, x_{i,\text{Split System}} + 3368\, x_{i,\text{Ductless System}} + 2135\, x_{i,\text{Central AC}} + 1046\, x_{i,\text{Hybrid AC}} + 4030\, x_{i,\text{Geothermal AC}} + 3761\, x_{i,\text{Smart AC}} + 3523\, x_{i,\text{Evaporative Cooler}} + 1701\, x_{i,\text{Package Unit}} \Big)
$$

**Subject to (for each storage area $i$):**

For $i=1$:
$$
114\, x_{1,\text{Window Unit}} + 200\, x_{1,\text{Portable Unit}} + 106\, x_{1,\text{Split System}} + 256\, x_{1,\text{Ductless System}} + 268\, x_{1,\text{Central AC}} + 185\, x_{1,\text{Hybrid AC}} + 299\, x_{1,\text{Geothermal AC}} + 131\, x_{1,\text{Smart AC}} + 139\, x_{1,\text{Evaporative Cooler}} + 105\, x_{1,\text{Package Unit}} \leq 1083
$$

For $i=2$:
$$
114\, x_{2,\text{Window Unit}} + 200\, x_{2,\text{Portable Unit}} + 106\, x_{2,\text{Split System}} + 256\, x_{2,\text{Ductless System}} + 268\, x_{2,\text{Central AC}} + 185\, x_{2,\text{Hybrid AC}} + 299\, x_{2,\text{Geothermal AC}} + 131\, x_{2,\text{Smart AC}} + 139\, x_{2,\text{Evaporative Cooler}} + 105\, x_{2,\text{Package Unit}} \leq 1840
$$

For $i=3$:
$$
114\, x_{3,\text{Window Unit}} + 200\, x_{3,\text{Portable Unit}} + 106\, x_{3,\text{Split System}} + 256\, x_{3,\text{Ductless System}} + 268\, x_{3,\text{Central AC}} + 185\, x_{3,\text{Hybrid AC}} + 299\, x_{3,\text{Geothermal AC}} + 131\, x_{3,\text{Smart AC}} + 139\, x_{3,\text{Evaporative Cooler}} + 105\, x_{3,\text{Package Unit}} \leq 770
$$

For $i=4$:
$$
114\, x_{4,\text{Window Unit}} + 200\, x_{4,\text{Portable Unit}} + 106\, x_{4,\text{Split System}} + 256\, x_{4,\text{Ductless System}} + 268\, x_{4,\text{Central AC}} + 185\, x_{4,\text{Hybrid AC}} + 299\, x_{4,\text{Geothermal AC}} + 131\, x_{4,\text{Smart AC}} + 139\, x_{4,\text{Evaporative Cooler}} + 105\, x_{4,\text{Package Unit}} \leq 1299
$$

For $i=5$:
$$
114\, x_{5,\text{Window Unit}} + 200\, x_{5,\text{Portable Unit}} + 106\, x_{5,\text{Split System}} + 256\, x_{5,\text{Ductless System}} + 268\, x_{5,\text{Central AC}} + 185\, x_{5,\text{Hybrid AC}} + 299\, x_{5,\text{Geothermal AC}} + 131\, x_{5,\text{Smart AC}} + 139\, x_{5,\text{Evaporative Cooler}} + 105\, x_{5,\text{Package Unit}} \leq 1259
$$

For $i=6$:
$$
114\, x_{6,\text{Window Unit}} + 200\, x_{6,\text{Portable Unit}} + 106\, x_{6,\text{Split System}} + 256\, x_{6,\text{Ductless System}} + 268\, x_{6,\text{Central AC}} + 185\, x_{6,\text{Hybrid AC}} + 299\, x_{6,\text{Geothermal AC}} + 131\, x_{6,\text{Smart AC}} + 139\, x_{6,\text{Evaporative Cooler}} + 105\, x_{6,\text{Package Unit}} \leq 543
$$

For $i=7$:
$$
114\, x_{7,\text{Window Unit}} + 200\, x_{7,\text{Portable Unit}} + 106\, x_{7,\text{Split System}} + 256\, x_{7,\text{Ductless System}} + 268\, x_{7,\text{Central AC}} + 185\, x_{7,\text{Hybrid AC}} + 299\, x_{7,\text{Geothermal AC}} + 131\, x_{7,\text{Smart AC}} + 139\, x_{7,\text{Evaporative Cooler}} + 105\, x_{7,\text{Package Unit}} \leq 1831
$$

For $i=8$:
$$
114\, x_{8,\text{Window Unit}} + 200\, x_{8,\text{Portable Unit}} + 106\, x_{8,\text{Split System}} + 256\, x_{8,\text{Ductless System}} + 268\, x_{8,\text{Central AC}} + 185\, x_{8,\text{Hybrid AC}} + 299\, x_{8,\text{Geothermal AC}} + 131\, x_{8,\text{Smart AC}} + 139\, x_{8,\text{Evaporative Cooler}} + 105\, x_{8,\text{Package Unit}} \leq 855
$$

For $i=9$:
$$
114\, x_{9,\text{Window Unit}} + 200\, x_{9,\text{Portable Unit}} + 106\, x_{9,\text{Split System}} + 256\, x_{9,\text{Ductless System}} + 268\, x_{9,\text{Central AC}} + 185\, x_{9,\text{Hybrid AC}} + 299\, x_{9,\text{Geothermal AC}} + 131\, x_{9,\text{Smart AC}} + 139\, x_{9,\text{Evaporative Cooler}} + 105\, x_{9,\text{Package Unit}} \leq 619
$$

For $i=10$:
$$
114\, x_{10,\text{Window Unit}} + 200\, x_{10,\text{Portable Unit}} + 106\, x_{10,\text{Split System}} + 256\, x_{10,\text{Ductless System}} + 268\, x_{10,\text{Central AC}} + 185\, x_{10,\text{Hybrid AC}} + 299\, x_{10,\text{Geothermal AC}} + 131\, x_{10,\text{Smart AC}} + 139\, x_{10,\text{Evaporative Cooler}} + 105\, x_{10,\text{Package Unit}} \leq 637
$$

For $i=11$:
$$
114\, x_{11,\text{Window Unit}} + 200\, x_{11,\text{Portable Unit}} + 106\, x_{11,\text{Split System}} + 256\, x_{11,\text{Ductless System}} + 268\, x_{11,\text{Central AC}} + 185\, x_{11,\text{Hybrid AC}} + 299\, x_{11,\text{Geothermal AC}} + 131\, x_{11,\text{Smart AC}} + 139\, x_{11,\text{Evaporative Cooler}} + 105\, x_{11,\text{Package Unit}} \leq 935
$$

For $i=12$:
$$
114\, x_{12,\text{Window Unit}} + 200\, x_{12,\text{Portable Unit}} + 106\, x_{12,\text{Split System}} + 256\, x_{12,\text{Ductless System}} + 268\, x_{12,\text{Central AC}} + 185\, x_{12,\text{Hybrid AC}} + 299\, x_{12,\text{Geothermal AC}} + 131\, x_{12,\text{Smart AC}} + 139\, x_{12,\text{Evaporative Cooler}} + 105\, x_{12,\text{Package Unit}} \leq 626
$$

For $i=13$:
$$
114\, x_{13,\text{Window Unit}} + 200\, x_{13,\text{Portable Unit}} + 106\, x_{13,\text{Split System}} + 256\, x_{13,\text{Ductless System}} + 268\, x_{13,\text{Central AC}} + 185\, x_{13,\text{Hybrid AC}} + 299\, x_{13,\text{Geothermal AC}} + 131\, x_{13,\text{Smart AC}} + 139\, x_{13,\text{Evaporative Cooler}} + 105\, x_{13,\text{Package Unit}} \leq 1457
$$

For $i=14$:
$$
114\, x_{14,\text{Window Unit}} + 200\, x_{14,\text{Portable Unit}} + 106\, x_{14,\text{Split System}} + 256\, x_{14,\text{Ductless System}} + 268\, x_{14,\text{Central AC}} + 185\, x_{14,\text{Hybrid AC}} + 299\, x_{14,\text{Geothermal AC}} + 131\, x_{14,\text{Smart AC}} + 139\, x_{14,\text{Evaporative Cooler}} + 105\, x_{14,\text{Package Unit}} \leq 1198
$$

For $i=15$:
$$
114\, x_{15,\text{Window Unit}} + 200\, x_{15,\text{Portable Unit}} + 106\, x_{15,\text{Split System}} + 256\, x_{15,\text{Ductless System}} + 268\, x_{15,\text{Central AC}} + 185\, x_{15,\text{Hybrid AC}} + 299\, x_{15,\text{Geothermal AC}} + 131\, x_{15,\text{Smart AC}} + 139\, x_{15,\text{Evaporative Cooler}} + 105\, x_{15,\text{Package Unit}} \leq 837
$$

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,15\},\ j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}
$$

**All coefficients, identifiers, and constraints are as retrieved and in source order.**