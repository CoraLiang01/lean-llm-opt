##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether warehouse $i$ is activated (binary).

##### Parameters

- Warehouses $I = \{\text{S1}, \text{S2}, \text{S3}\}$
- Musicians/Bands $J = \{\text{C1}, \text{C2}, \text{C3}\}$

- Demands:
  - $d_{\text{C1}} = 1083$
  - $d_{\text{C2}} = 776$
  - $d_{\text{C3}} = 16214$

- Fixed costs:
  - $f_{\text{S1}} = 102.33$
  - $f_{\text{S2}} = 94.92$
  - $f_{\text{S3}} = 91.83$

- Transportation costs per unit:
  - $c_{\text{S1},\text{C1}} = 1506.22$, $c_{\text{S1},\text{C2}} = 70.9$, $c_{\text{S1},\text{C3}} = 8.44$
  - $c_{\text{S2},\text{C1}} = 1732.65$, $c_{\text{S2},\text{C2}} = 1780.72$, $c_{\text{S2},\text{C3}} = 567.44$
  - $c_{\text{S3},\text{C1}} = 115.66$, $c_{\text{S3},\text{C2}} = 100.76$, $c_{\text{S3},\text{C3}} = 64.68$

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

2. Warehouse activation: shipments only from active warehouses:
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. Variable domains:
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Full Model Data

- $I = \{\text{S1}, \text{S2}, \text{S3}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}\}$
- $d_{\text{C1}} = 1083$, $d_{\text{C2}} = 776$, $d_{\text{C3}} = 16214$
- $f_{\text{S1}} = 102.33$, $f_{\text{S2}} = 94.92$, $f_{\text{S3}} = 91.83$
- $c_{ij}$ matrix:

|        | C1      | C2      | C3      |
|--------|---------|---------|---------|
| S1     | 1506.22 | 70.9    | 8.44    |
| S2     | 1732.65 | 1780.72 | 567.44  |
| S3     | 115.66  | 100.76  | 64.68   |

- $M = 18073$

##### Model Summary

\[
\begin{align*}
\min\ & \sum_{i \in \{\text{S1},\text{S2},\text{S3}\}} \sum_{j \in \{\text{C1},\text{C2},\text{C3}\}} c_{ij} x_{ij} + \sum_{i \in \{\text{S1},\text{S2},\text{S3}\}} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in \{\text{S1},\text{S2},\text{S3}\}} x_{ij} = d_j, \quad \forall j \in \{\text{C1},\text{C2},\text{C3}\} \\
& \sum_{j \in \{\text{C1},\text{C2},\text{C3}\}} x_{ij} \leq 18073\, y_i, \quad \forall i \in \{\text{S1},\text{S2},\text{S3}\} \\
& x_{ij} \geq 0,\quad y_i \in \{0,1\}
\end{align*}
\]