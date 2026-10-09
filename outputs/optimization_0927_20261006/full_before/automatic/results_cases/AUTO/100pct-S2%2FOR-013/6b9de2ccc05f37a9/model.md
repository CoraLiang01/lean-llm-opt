Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName $j$) to be placed in storage area $i$ (StorageID $i$). All $x_{ij}$ are integer and nonnegative.

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

  With capacities:

  - $C_1 = 1083$
  - $C_2 = 1840$
  - $C_3 = 770$
  - $C_4 = 1299$
  - $C_5 = 1259$
  - $C_6 = 543$
  - $C_7 = 1831$
  - $C_8 = 855$
  - $C_9 = 619$
  - $C_{10} = 637$
  - $C_{11} = 935$
  - $C_{12} = 626$
  - $C_{13} = 1457$
  - $C_{14} = 1198$
  - $C_{15} = 837$

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

Objective:
\[
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of product $j$.

Subject to, for each storage area $i$ (StorageID):

\[
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
\]
where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of storage area $i$.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

**Explicitly:**

Let $J$ be the set of product names in the order above, and $I$ be the set of StorageIDs in the order above.

- For all $i \in I$:
  \[
  \sum_{j \in J} w_j x_{ij} \leq C_i
  \]
- For all $i \in I$, $j \in J$:
  \[
  x_{ij} \in \mathbb{Z}_{\geq 0}
  \]

- Objective:
  \[
  \max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
  \]

Where:

- $v_j$ and $w_j$ are as follows (in source order):

  | ProductName           | $v_j$ | $w_j$ |
  |-----------------------|-------|-------|
  | Window Unit           | 4811  | 114   |
  | Portable Unit         | 1130  | 200   |
  | Split System          | 1611  | 106   |
  | Ductless System       | 3368  | 256   |
  | Central AC            | 2135  | 268   |
  | Hybrid AC             | 1046  | 185   |
  | Geothermal AC         | 4030  | 299   |
  | Smart AC              | 3761  | 131   |
  | Evaporative Cooler    | 3523  | 139   |
  | Package Unit          | 1701  | 105   |

And $C_i$ as above for each StorageID $i$.

**All variables, coefficients, and constraints are included as retrieved and required.**