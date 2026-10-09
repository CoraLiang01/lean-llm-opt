##### Decision Variables

$x_{ij} \geq 0$: Quantity of goods that musician or band $j \in J$ sources from warehouse $i \in I$ (continuous).
$y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise.

##### Parameters

- Warehouses $I = \{S1, S2, S3\}$
- Musicians/Bands $J = \{C1, C2, C3\}$

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

1. Demand satisfaction for each musician/band:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. Warehouse activation: No shipments from inactive warehouses:
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. Variable domains:
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Full Parameter Listing

- $I = \{S1, S2, S3\}$
- $J = \{C1, C2, C3\}$
- Demands: $d_{C1} = 1083$, $d_{C2} = 776$, $d_{C3} = 16214$
- Fixed costs: $f_{S1} = 102.33$, $f_{S2} = 94.92$, $f_{S3} = 91.83$
- Transportation costs:
  - $c_{S1,C1} = 1506.22$, $c_{S1,C2} = 70.9$, $c_{S1,C3} = 8.44$
  - $c_{S2,C1} = 1732.65$, $c_{S2,C2} = 1780.72$, $c_{S2,C3} = 567.44$
  - $c_{S3,C1} = 115.66$, $c_{S3,C2} = 100.76$, $c_{S3,C3} = 64.68$
- $M = 18073$

##### Model Summary

\[
\begin{align*}
\min \quad & \sum_{i \in \{S1,S2,S3\}} \sum_{j \in \{C1,C2,C3\}} c_{ij} x_{ij} + \sum_{i \in \{S1,S2,S3\}} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in \{S1,S2,S3\}} x_{ij} = d_j, \quad \forall j \in \{C1,C2,C3\} \\
& \sum_{j \in \{C1,C2,C3\}} x_{ij} \leq 18073\, y_i, \quad \forall i \in \{S1,S2,S3\} \\
& x_{ij} \geq 0, \quad y_i \in \{0,1\}
\end{align*}
\]