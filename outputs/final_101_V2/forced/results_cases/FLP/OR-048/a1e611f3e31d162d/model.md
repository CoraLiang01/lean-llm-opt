##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of air conditioner type $j$ placed in storage area $i$, for all storage areas $i \in I$ and product types $j \in J$.

##### Parameters

- $I = \{1, 2, 3, \ldots, 15\}$ (storage area indices)
- $J = \{$
  1: Window Unit,
  2: Portable Unit,
  3: Split System,
  4: Ductless System,
  5: Central AC,
  6: Hybrid AC,
  7: Geothermal AC,
  8: Smart AC,
  9: Evaporative Cooler,
  10: Package Unit
$\}$ (product type indices)

- Storage area capacities:
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

- Product values and weights:
  - $v_1 = 4811$, $w_1 = 114$ (Window Unit)
  - $v_2 = 1130$, $w_2 = 200$ (Portable Unit)
  - $v_3 = 1611$, $w_3 = 106$ (Split System)
  - $v_4 = 3368$, $w_4 = 256$ (Ductless System)
  - $v_5 = 2135$, $w_5 = 268$ (Central AC)
  - $v_6 = 1046$, $w_6 = 185$ (Hybrid AC)
  - $v_7 = 4030$, $w_7 = 299$ (Geothermal AC)
  - $v_8 = 3761$, $w_8 = 131$ (Smart AC)
  - $v_9 = 3523$, $w_9 = 139$ (Evaporative Cooler)
  - $v_{10} = 1701$, $w_{10} = 105$ (Package Unit)

##### Objective Function

\[
\max \sum_{i=1}^{15} \sum_{j=1}^{10} v_j x_{ij}
\]

##### Constraints

1. Storage area capacity constraints:
   \[
   \sum_{j=1}^{10} w_j x_{ij} \leq C_i, \quad \forall i = 1, \ldots, 15
   \]

2. Integer and nonnegativity constraints:
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 15,\; j = 1, \ldots, 10
   \]

##### Full Parameter Listing

- Storage areas $I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$
- Product types $J = \{1,2,3,4,5,6,7,8,9,10\}$

- Capacities:
  - $C = [1083, 1840, 770, 1299, 1259, 543, 1831, 855, 619, 637, 935, 626, 1457, 1198, 837]$

- Values:
  - $v = [4811, 1130, 1611, 3368, 2135, 1046, 4030, 3761, 3523, 1701]$

- Weights:
  - $w = [114, 200, 106, 256, 268, 185, 299, 131, 139, 105]$

##### Model Summary

\[
\begin{align*}
\max\quad & \sum_{i=1}^{15} \sum_{j=1}^{10} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{10} w_j x_{ij} \leq C_i, \quad \forall i = 1, \ldots, 15 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 15,\; j = 1, \ldots, 10
\end{align*}
\]