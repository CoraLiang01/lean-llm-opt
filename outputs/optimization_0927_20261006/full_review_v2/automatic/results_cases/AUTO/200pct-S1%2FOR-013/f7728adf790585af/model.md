Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are nonnegative integers.

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

  Each StorageID $i$ has capacity $C_i$:

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

Let $V_j$ be the Value and $W_j$ the Weight (size) of product $j$.

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,15\},\ j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}
$$

**Objective:**

$$
\max \sum_{i=1}^{15} \sum_{j} V_j \cdot x_{ij}
$$

where $V_j$ is as above for each ProductName.

**Constraints:**

For each storage area $i$ (StorageID):

$$
\sum_{j} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

where $W_j$ is as above for each ProductName, and $C_i$ is the capacity for StorageID $i$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Explicit Data Used:**

- Storage areas (StorageID) and their capacities (Capacity), in source order:

  1: 1083  
  2: 1840  
  3: 770  
  4: 1299  
  5: 1259  
  6: 543  
  7: 1831  
  8: 855  
  9: 619  
  10: 637  
  11: 935  
  12: 626  
  13: 1457  
  14: 1198  
  15: 837  

- Air conditioner types (ProductName), their Value, and Weight (size), in source order:

  - Window Unit: Value = 4811, Weight = 114
  - Portable Unit: Value = 1130, Weight = 200
  - Split System: Value = 1611, Weight = 106
  - Ductless System: Value = 3368, Weight = 256
  - Central AC: Value = 2135, Weight = 268
  - Hybrid AC: Value = 1046, Weight = 185
  - Geothermal AC: Value = 4030, Weight = 299
  - Smart AC: Value = 3761, Weight = 131
  - Evaporative Cooler: Value = 3523, Weight = 139
  - Package Unit: Value = 1701, Weight = 105

---

**Summary:**

Maximize total value of air conditioners allocated to storage areas, subject to each area's capacity, using integer variables $x_{ij}$ for each (StorageID, ProductName) pair, with the above data and constraints.