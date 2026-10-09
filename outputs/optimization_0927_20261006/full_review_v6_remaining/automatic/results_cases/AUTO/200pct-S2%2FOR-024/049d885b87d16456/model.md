##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods that musician or band $j \in J$ sources from warehouse $i \in I$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated (operational), 0 otherwise (binary).

##### Parameters

- Warehouses (facilities): $I = \{S1, S2, S3\}$
- Musicians/Bands (customers): $J = \{C1, C2, C3\}$

- Demand:
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

- Total demand: $M = 1083 + 776 + 16214 = 18073$ (used as a sufficiently large upper bound)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:** Each musician/band receives exactly their demand.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Warehouse activation:** No goods can be shipped from an inactive warehouse.
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Sets and Data

- $I = \{S1, S2, S3\}$
- $J = \{C1, C2, C3\}$
- $d_{C1} = 1083$, $d_{C2} = 776$, $d_{C3} = 16214$
- $f_{S1} = 102.33$, $f_{S2} = 94.92$, $f_{S3} = 91.83$
- $c_{ij}$ matrix:

|        | C1      | C2      | C3      |
|--------|---------|---------|---------|
| S1     | 1506.22 | 70.9    | 8.44    |
| S2     | 1732.65 | 1780.72 | 567.44  |
| S3     | 115.66  | 100.76  | 64.68   |

- $M = 18073$

##### Full Model

\[
\begin{align*}
\min\ & \sum_{i \in \{S1, S2, S3\}} \sum_{j \in \{C1, C2, C3\}} c_{ij} x_{ij} + \sum_{i \in \{S1, S2, S3\}} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in \{S1, S2, S3\}} x_{ij} = d_j, \quad \forall j \in \{C1, C2, C3\} \\
& \sum_{j \in \{C1, C2, C3\}} x_{ij} \leq 18073\, y_i, \quad \forall i \in \{S1, S2, S3\} \\
& x_{ij} \geq 0,\quad \forall i \in \{S1, S2, S3\},\ j \in \{C1, C2, C3\} \\
& y_i \in \{0,1\},\quad \forall i \in \{S1, S2, S3\}
\end{align*}
\]

Where all parameters and sets are as listed above.