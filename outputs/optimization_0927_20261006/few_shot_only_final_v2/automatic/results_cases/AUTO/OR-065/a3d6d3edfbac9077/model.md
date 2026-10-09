##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous)
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise

##### Parameters

- $I = \{S1, S2, S3\}$ (warehouses)
- $J = \{C1, C2, C3\}$ (musicians/bands)
- Demands: $d_{C1} = 1083$, $d_{C2} = 776$, $d_{C3} = 16214$
- Fixed costs: $f_{S1} = 102.33$, $f_{S2} = 94.92$, $f_{S3} = 91.83$
- Transportation costs $c_{ij}$:

|         | C1      | C2      | C3      |
|---------|---------|---------|---------|
| S1      | 1506.22 | 70.90   | 8.44    |
| S2      | 1732.65 | 1780.72 | 567.44  |
| S3      | 115.66  | 100.76  | 64.68   |

- $M = 18073$

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
   That is,
   - $x_{S1,C1} + x_{S2,C1} + x_{S3,C1} = 1083$
   - $x_{S1,C2} + x_{S2,C2} + x_{S3,C2} = 776$
   - $x_{S1,C3} + x_{S2,C3} + x_{S3,C3} = 16214$

2. **Warehouse activation:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   That is,
   - $x_{S1,C1} + x_{S1,C2} + x_{S1,C3} \leq 18073\, y_{S1}$
   - $x_{S2,C1} + x_{S2,C2} + x_{S2,C3} \leq 18073\, y_{S2}$
   - $x_{S3,C1} + x_{S3,C2} + x_{S3,C3} \leq 18073\, y_{S3}$

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
\min\quad & 1506.22\,x_{S1,C1} + 70.90\,x_{S1,C2} + 8.44\,x_{S1,C3} \\
         & + 1732.65\,x_{S2,C1} + 1780.72\,x_{S2,C2} + 567.44\,x_{S2,C3} \\
         & + 115.66\,x_{S3,C1} + 100.76\,x_{S3,C2} + 64.68\,x_{S3,C3} \\
         & + 102.33\,y_{S1} + 94.92\,y_{S2} + 91.83\,y_{S3} \\[2ex]
\text{s.t.}\quad
& x_{S1,C1} + x_{S2,C1} + x_{S3,C1} = 1083 \\
& x_{S1,C2} + x_{S2,C2} + x_{S3,C2} = 776 \\
& x_{S1,C3} + x_{S2,C3} + x_{S3,C3} = 16214 \\[2ex]
& x_{S1,C1} + x_{S1,C2} + x_{S1,C3} \leq 18073\, y_{S1} \\
& x_{S2,C1} + x_{S2,C2} + x_{S2,C3} \leq 18073\, y_{S2} \\
& x_{S3,C1} + x_{S3,C2} + x_{S3,C3} \leq 18073\, y_{S3} \\[2ex]
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]

Where:
- $I = \{S1, S2, S3\}$
- $J = \{C1, C2, C3\}$
- $d_{C1} = 1083$, $d_{C2} = 776$, $d_{C3} = 16214$
- $f_{S1} = 102.33$, $f_{S2} = 94.92$, $f_{S3} = 91.83$
- $c_{S1,C1} = 1506.22$, $c_{S1,C2} = 70.90$, $c_{S1,C3} = 8.44$
- $c_{S2,C1} = 1732.65$, $c_{S2,C2} = 1780.72$, $c_{S2,C3} = 567.44$
- $c_{S3,C1} = 115.66$, $c_{S3,C2} = 100.76$, $c_{S3,C3} = 64.68$
- $M = 18073$