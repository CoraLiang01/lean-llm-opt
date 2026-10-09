##### Sets

Let $I = \{1,2,\ldots,15\}$ be the set of storage areas (StorageID from 1 to 15).

Let $J = \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$ be the set of air conditioner types.

##### Parameters

For each storage area $i \in I$:

- $C_i$ = capacity of storage area $i$.

For each product $j \in J$:

- $v_j$ = value per unit of product $j$.
- $w_j$ = weight (size) per unit of product $j$.

From the data:

- $C_1 = 1083$, $C_2 = 1840$, $C_3 = 770$, $C_4 = 1299$, $C_5 = 1259$, $C_6 = 543$, $C_7 = 1831$, $C_8 = 855$, $C_9 = 619$, $C_{10} = 637$, $C_{11} = 935$, $C_{12} = 626$, $C_{13} = 1457$, $C_{14} = 1198$, $C_{15} = 837$.

- Product data:

| $j$                  | $v_j$ | $w_j$ |
|----------------------|-------|-------|
| Window Unit          | 4811  | 114   |
| Portable Unit        | 1130  | 200   |
| Split System         | 1611  | 106   |
| Ductless System      | 3368  | 256   |
| Central AC           | 2135  | 268   |
| Hybrid AC            | 1046  | 185   |
| Geothermal AC        | 4030  | 299   |
| Smart AC             | 3761  | 131   |
| Evaporative Cooler   | 3523  | 139   |
| Package Unit         | 1701  | 105   |

##### Decision Variables

For each $i \in I$, $j \in J$:

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed in storage area $i$.

##### Objective

Maximize total value:

$$
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
$$

##### Constraints

1. Capacity constraints for each storage area:
   $$
   \sum_{j \in J} w_j x_{ij} \leq C_i \qquad \forall i \in I
   $$

2. Nonnegativity and integrality:
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   $$

##### Retrieved Information

Storage areas and capacities:
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

Products:
- Window Unit: value 4811, weight 114
- Portable Unit: value 1130, weight 200
- Split System: value 1611, weight 106
- Ductless System: value 3368, weight 256
- Central AC: value 2135, weight 268
- Hybrid AC: value 1046, weight 185
- Geothermal AC: value 4030, weight 299
- Smart AC: value 3761, weight 131
- Evaporative Cooler: value 3523, weight 139
- Package Unit: value 1701, weight 105