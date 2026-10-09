##### Sets

- $I = \{1, 2, \ldots, 15\}$: set of storage areas (StorageID).
- $J = \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}$: set of air conditioner product types.

##### Parameters

- $C_i$: capacity of storage area $i$.

  $C_1 = 1083$, $C_2 = 1840$, $C_3 = 770$, $C_4 = 1299$, $C_5 = 1259$, $C_6 = 543$, $C_7 = 1831$, $C_8 = 855$, $C_9 = 619$, $C_{10} = 637$, $C_{11} = 935$, $C_{12} = 626$, $C_{13} = 1457$, $C_{14} = 1198$, $C_{15} = 837$

- $v_j$: value of one unit of product $j$.

  - $v_{\text{Window Unit}} = 4811$
  - $v_{\text{Portable Unit}} = 1130$
  - $v_{\text{Split System}} = 1611$
  - $v_{\text{Ductless System}} = 3368$
  - $v_{\text{Central AC}} = 2135$
  - $v_{\text{Hybrid AC}} = 1046$
  - $v_{\text{Geothermal AC}} = 4030$
  - $v_{\text{Smart AC}} = 3761$
  - $v_{\text{Evaporative Cooler}} = 3523$
  - $v_{\text{Package Unit}} = 1701$

- $w_j$: size (weight) of one unit of product $j$.

  - $w_{\text{Window Unit}} = 114$
  - $w_{\text{Portable Unit}} = 200$
  - $w_{\text{Split System}} = 106$
  - $w_{\text{Ductless System}} = 256$
  - $w_{\text{Central AC}} = 268$
  - $w_{\text{Hybrid AC}} = 185$
  - $w_{\text{Geothermal AC}} = 299$
  - $w_{\text{Smart AC}} = 131$
  - $w_{\text{Evaporative Cooler}} = 139$
  - $w_{\text{Package Unit}} = 105$

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed in storage area $i$.

##### Objective Function

\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

##### Constraints

1. **Capacity constraints for each storage area:**

   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I
   \]

2. **Nonnegativity and integrality:**

   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

###### Retrieved Information

- Storage areas and capacities:
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

- Product types, values, and sizes:
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