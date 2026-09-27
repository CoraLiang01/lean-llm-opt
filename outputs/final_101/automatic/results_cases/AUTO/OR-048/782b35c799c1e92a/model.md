Let $x_{ij}$ be the number of units of air conditioner type $j$ to be placed in storage area $i$. All $x_{ij}$ are integer and $\geq 0$.

Indices:
- $i$ indexes StorageID $\in \{1,2,\ldots,15\}$
- $j$ indexes ProductName $\in \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$

Parameters:
- $c_i$ = Capacity of storage area $i$
- $v_j$ = Value of air conditioner type $j$
- $w_j$ = Weight (size) of air conditioner type $j$

Data:

Storage Areas and Capacities:
- 1: $c_1 = 1083$
- 2: $c_2 = 1840$
- 3: $c_3 = 770$
- 4: $c_4 = 1299$
- 5: $c_5 = 1259$
- 6: $c_6 = 543$
- 7: $c_7 = 1831$
- 8: $c_8 = 855$
- 9: $c_9 = 619$
- 10: $c_{10} = 637$
- 11: $c_{11} = 935$
- 12: $c_{12} = 626$
- 13: $c_{13} = 1457$
- 14: $c_{14} = 1198$
- 15: $c_{15} = 837$

Air Conditioner Types, Values, and Weights:
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

Objective:
$$
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j x_{ij}
$$

Subject to:

For each storage area $i$:
$$
\sum_{j=1}^{10} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,15\}
$$

Integrality and nonnegativity:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,15\},\ j \in \{1,\ldots,10\}
$$

Where the mapping of $j$ to ProductName, $v_j$, and $w_j$ is as listed above.