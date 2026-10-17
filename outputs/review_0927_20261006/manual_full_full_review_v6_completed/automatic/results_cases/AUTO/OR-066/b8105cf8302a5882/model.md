##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Parameters

- $I = \{S1, S2\}$ (suppliers)
- $J = \{C1, C2\}$ (supermarkets)
- Demands:
  - $d_{C1} = 144$
  - $d_{C2} = 216$
- Fixed costs:
  - $f_{S1} = 105.97$
  - $f_{S2} = 85.31$
- Transportation costs:
  - $c_{S1,C1} = 2358.39$
  - $c_{S1,C2} = 1492.08$
  - $c_{S2,C1} = 0.07$
  - $c_{S2,C2} = 52.32$
- $M = \sum_{j \in J} d_j = 144 + 216 = 360$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Supermarket demand satisfaction:**  
   $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. **Supplier activation:**  
   $\sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I$

3. **Variable domains:**  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Explicit Data

- $I = \{S1, S2\}$
- $J = \{C1, C2\}$
- $d_{C1} = 144$, $d_{C2} = 216$
- $f_{S1} = 105.97$, $f_{S2} = 85.31$
- $c_{S1,C1} = 2358.39$, $c_{S1,C2} = 1492.08$
- $c_{S2,C1} = 0.07$, $c_{S2,C2} = 52.32$
- $M = 360$