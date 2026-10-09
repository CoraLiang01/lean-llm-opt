##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: Whether warehouse $i$ is activated (binary).

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
- Transportation costs $c_{ij}$:

|         | C1      | C2      | C3      |
|---------|---------|---------|---------|
| S1      | 1506.22 | 70.9    | 8.44    |
| S2      | 1732.65 | 1780.72 | 567.44  |
| S3      | 115.66  | 100.76  | 64.68   |

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
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Full Model Data

- $I = \{\text{S1}, \text{S2}, \text{S3}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}\}$
- $d_{\text{C1}} = 1083$, $d_{\text{C2}} = 776$, $d_{\text{C3}} = 16214$
- $f_{\text{S1}} = 102.33$, $f_{\text{S2}} = 94.92$, $f_{\text{S3}} = 91.83$
- $c_{ij}$ as in the table above
- $M = 18073$