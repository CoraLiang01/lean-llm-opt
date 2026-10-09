##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods that musician or band $j \in J$ sources from warehouse $i \in I$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is operational, 0 otherwise (binary).

##### Parameters

- $I = \{S1, S2, S3\}$ (warehouses)
- $J = \{C1, C2, C3\}$ (musicians/bands)
- Demands:
  - $d_{C1} = 1083$
  - $d_{C2} = 776$
  - $d_{C3} = 16214$
- Fixed costs:
  - $f_{S1} = 102.33$
  - $f_{S2} = 94.92$
  - $f_{S3} = 91.83$
- Transportation costs per unit:
  - $c_{S1,C1} = 1506.22$, $c_{S1,C2} = 70.9$, $c_{S1,C3} = 8.44$
  - $c_{S2,C1} = 1732.65$, $c_{S2,C2} = 1780.72$, $c_{S2,C3} = 567.44$
  - $c_{S3,C1} = 115.66$, $c_{S3,C2} = 100.76$, $c_{S3,C3} = 64.68$
- $M = \sum_{j \in J} d_j = 1083 + 776 + 16214 = 18073$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each musician/band $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Warehouse activation:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   (A warehouse can only ship goods if it is operational.)
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Full Data Used

- Warehouses: $I = \{S1, S2, S3\}$
- Musicians/Bands: $J = \{C1, C2, C3\}$
- Demands: $d_{C1} = 1083$, $d_{C2} = 776$, $d_{C3} = 16214$
- Fixed costs: $f_{S1} = 102.33$, $f_{S2} = 94.92$, $f_{S3} = 91.83$
- Transportation costs:
  - $c_{S1,C1} = 1506.22$, $c_{S1,C2} = 70.9$, $c_{S1,C3} = 8.44$
  - $c_{S2,C1} = 1732.65$, $c_{S2,C2} = 1780.72$, $c_{S2,C3} = 567.44$
  - $c_{S3,C1} = 115.66$, $c_{S3,C2} = 100.76$, $c_{S3,C3} = 64.68$
- $M = 18073$