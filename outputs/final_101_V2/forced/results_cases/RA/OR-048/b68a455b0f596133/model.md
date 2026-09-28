Let $x_{ij}$ be the number of units of air conditioner type $j$ to be placed in storage area $i$. All $x_{ij}$ are integer and $\geq 0$.

Indices:
- $i$ indexes StorageID $\in \{1,2,\ldots,15\}$
- $j$ indexes ProductName $\in \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$

Parameters:
- $v_j$ = Value of product $j$ (see below)
- $w_j$ = Weight of product $j$ (see below)
- $C_i$ = Capacity of storage area $i$ (see below)

Objective:
$$
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j x_{ij}
$$

Subject to, for each storage area $i$:
$$
\sum_{j=1}^{10} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

Integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

Where:

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

Product Types, Values, and Weights:
- Window Unit: $v_1 = 4811$, $w_1 = 114$
- Portable Unit: $v_2 = 1130$, $w_2 = 200$
- Split System: $v_3 = 1611$, $w_3 = 106$
- Ductless System: $v_4 = 3368$, $w_4 = 256$
- Central AC: $v_5 = 2135$, $w_5 = 268$
- Hybrid AC: $v_6 = 1046$, $w_6 = 185$
- Geothermal AC: $v_7 = 4030$, $w_7 = 299$
- Smart AC: $v_8 = 3761$, $w_8 = 131$
- Evaporative Cooler: $v_9 = 3523$, $w_9 = 139$
- Package Unit: $v_{10} = 1701$, $w_{10} = 105$