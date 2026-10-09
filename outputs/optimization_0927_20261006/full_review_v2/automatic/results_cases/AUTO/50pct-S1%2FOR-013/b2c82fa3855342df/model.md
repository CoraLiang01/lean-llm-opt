Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName $j$) to be placed in storage area $i$ (StorageID $i$). All $x_{ij}$ are integer and $\geq 0$.

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
  |----------------------|-------|--------|
  | Window Unit          | 4811  | 114    |
  | Portable Unit        | 1130  | 200    |
  | Split System         | 1611  | 106    |
  | Ductless System      | 3368  | 256    |
  | Central AC           | 2135  | 268    |
  | Hybrid AC            | 1046  | 185    |
  | Geothermal AC        | 4030  | 299    |
  | Smart AC             | 3761  | 131    |
  | Evaporative Cooler   | 3523  | 139    |
  | Package Unit         | 1701  | 105    |

**Mathematical Model:**

Maximize total value:
$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} v_j \cdot x_{ij}
$$

where $v_j$ is the Value for product $j$.

Subject to, for each storage area $i$ (using the source order):

- Capacity constraints:
$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$
where $w_j$ is the Weight for product $j$, and $C_i$ is the Capacity for StorageID $i$.

- Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

**Explicit Data:**

- Storage areas and capacities (in source order):

  1. StorageID 1: Capacity 1083
  2. StorageID 2: Capacity 1840
  3. StorageID 3: Capacity 770
  4. StorageID 4: Capacity 1299
  5. StorageID 5: Capacity 1259
  6. StorageID 6: Capacity 543
  7. StorageID 7: Capacity 1831
  8. StorageID 8: Capacity 855
  9. StorageID 9: Capacity 619
  10. StorageID 10: Capacity 637
  11. StorageID 11: Capacity 935
  12. StorageID 12: Capacity 626
  13. StorageID 13: Capacity 1457
  14. StorageID 14: Capacity 1198
  15. StorageID 15: Capacity 837

- Air conditioner types, values, and weights (in source order):

  1. Window Unit: Value 4811, Weight 114
  2. Portable Unit: Value 1130, Weight 200
  3. Split System: Value 1611, Weight 106
  4. Ductless System: Value 3368, Weight 256
  5. Central AC: Value 2135, Weight 268
  6. Hybrid AC: Value 1046, Weight 185
  7. Geothermal AC: Value 4030, Weight 299
  8. Smart AC: Value 3761, Weight 131
  9. Evaporative Cooler: Value 3523, Weight 139
  10. Package Unit: Value 1701, Weight 105

**Decision variables:**

$x_{ij}$ = number of units of air conditioner type $j$ (as listed above) to be placed in storage area $i$ (as listed above), for all $i = 1,\ldots,15$ and all $j = 1,\ldots,10$.

All $x_{ij} \in \mathbb{Z}_{\geq 0}$.