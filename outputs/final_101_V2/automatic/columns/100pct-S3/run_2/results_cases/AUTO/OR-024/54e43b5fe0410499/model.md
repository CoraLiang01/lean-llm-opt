##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Parameters

- $I = \{S1, S2, S3\}$ (warehouses)
- $J = \{C1, C2, C3\}$ (musicians/bands)

- Fixed costs per warehouse (current period):
  - $f_{S1} = 102.33$
  - $f_{S2} = 94.92$
  - $f_{S3} = 91.83$

- Demand per musician/band (current period):
  - $d_{C1} = 1083$
  - $d_{C2} = 776$
  - $d_{C3} = 16214$

- Transportation cost per unit (current period):

|         | C1      | C2      | C3      |
|---------|---------|---------|---------|
| S1      | 1506.22 | 70.9    | 8.44    |
| S2      | 1732.65 | 1780.72 | 567.44  |
| S3      | 115.66  | 100.76  | 64.68   |

Let $c_{ij}$ denote the transportation cost per unit from warehouse $i$ to musician/band $j$ as above.

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
   where $M = \sum_{j \in J} d_j = 1083 + 776 + 16214 = 18073$ (a valid upper bound since no explicit capacity is given).

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Complete Mathematical Model

\[
\begin{align*}
\min\quad & \sum_{i \in \{S1, S2, S3\}} \sum_{j \in \{C1, C2, C3\}} c_{ij} x_{ij} + \sum_{i \in \{S1, S2, S3\}} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in \{S1, S2, S3\}} x_{ij} = d_j \quad \forall j \in \{C1, C2, C3\} \\
& \sum_{j \in \{C1, C2, C3\}} x_{ij} \leq 18073\, y_i \quad \forall i \in \{S1, S2, S3\} \\
& x_{ij} \geq 0 \quad \forall i \in \{S1, S2, S3\},\, j \in \{C1, C2, C3\} \\
& y_i \in \{0,1\} \quad \forall i \in \{S1, S2, S3\}
\end{align*}
\]

where

- $f_{S1} = 102.33$, $f_{S2} = 94.92$, $f_{S3} = 91.83$
- $d_{C1} = 1083$, $d_{C2} = 776$, $d_{C3} = 16214$
- $c_{S1,C1} = 1506.22$, $c_{S1,C2} = 70.9$, $c_{S1,C3} = 8.44$
- $c_{S2,C1} = 1732.65$, $c_{S2,C2} = 1780.72$, $c_{S2,C3} = 567.44$
- $c_{S3,C1} = 115.66$, $c_{S3,C2} = 100.76$, $c_{S3,C3} = 64.68$
- $M = 18073$

All identifiers, coefficients, and matrix axes are preserved as in the source data.