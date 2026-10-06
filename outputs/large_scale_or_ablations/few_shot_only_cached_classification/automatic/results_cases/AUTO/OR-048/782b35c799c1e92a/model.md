Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

Parameters:
- $S$ = set of storage areas (StorageID): $\{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$
- $P$ = set of air conditioner types (ProductName): 
  - Window Unit
  - Portable Unit
  - Split System
  - Ductless System
  - Central AC
  - Hybrid AC
  - Geothermal AC
  - Smart AC
  - Evaporative Cooler
  - Package Unit
- $c_i$ = Capacity of storage area $i$ (from capacity.csv)
- $v_j$ = Value of air conditioner type $j$ (from products.csv)
- $w_j$ = Weight (size) of air conditioner type $j$ (from products.csv)

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

Subject to:

For each storage area $i \in S$:
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i
\]

For all $i \in S$, $j \in P$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

Where:

- Storage areas and their capacities:
    - 1: 1083
    - 2: 1840
    - 3: 770
    - 4: 1299
    - 5: 1259
    - 6: 543
    - 7: 1831
    - 8: 855
    - 9: 619
    - 10: 637
    - 11: 935
    - 12: 626
    - 13: 1457
    - 14: 1198
    - 15: 837

- Air conditioner types, values, and weights:
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

All variables $x_{ij}$ are integer and nonnegative.