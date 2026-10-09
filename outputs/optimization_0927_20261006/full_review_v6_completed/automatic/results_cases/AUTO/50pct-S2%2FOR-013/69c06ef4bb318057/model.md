Let $x_{ij}$ be the number of units of air conditioner type $j$ (corresponding to ProductName in products.csv) to be placed in storage area $i$ (corresponding to StorageID in capacity.csv). All $x_{ij}$ are nonnegative integers.

Define:
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

Let:
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight of product $j$ (from products.csv)
- $C_i$ = Capacity of storage area $i$ (from capacity.csv)

Parameters (from the data):

Storage Areas and Capacities:
- StorageID 1: $C_1 = 1083$
- StorageID 2: $C_2 = 1840$
- StorageID 3: $C_3 = 770$
- StorageID 4: $C_4 = 1299$
- StorageID 5: $C_5 = 1259$
- StorageID 6: $C_6 = 543$
- StorageID 7: $C_7 = 1831$
- StorageID 8: $C_8 = 855$
- StorageID 9: $C_9 = 619$
- StorageID 10: $C_{10} = 637$
- StorageID 11: $C_{11} = 935$
- StorageID 12: $C_{12} = 626$
- StorageID 13: $C_{13} = 1457$
- StorageID 14: $C_{14} = 1198$
- StorageID 15: $C_{15} = 837$

Air Conditioner Types, Values, and Weights:
- Window Unit: $v = 4811$, $w = 114$
- Portable Unit: $v = 1130$, $w = 200$
- Split System: $v = 1611$, $w = 106$
- Ductless System: $v = 3368$, $w = 256$
- Central AC: $v = 2135$, $w = 268$
- Hybrid AC: $v = 1046$, $w = 185$
- Geothermal AC: $v = 4030$, $w = 299$
- Smart AC: $v = 3761$, $w = 131$
- Evaporative Cooler: $v = 3523$, $w = 139$
- Package Unit: $v = 1701$, $w = 105$

Model:

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

Subject to, for each storage area $i \in S$:
\[
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
\]

Where:
- $x_{ij}$ = number of units of air conditioner type $j$ placed in storage area $i$
- $v_j$ = value of air conditioner type $j$ (see above)
- $w_j$ = weight (size) of air conditioner type $j$ (see above)
- $C_i$ = capacity of storage area $i$ (see above)

All parameters and indices are as retrieved and in original order.