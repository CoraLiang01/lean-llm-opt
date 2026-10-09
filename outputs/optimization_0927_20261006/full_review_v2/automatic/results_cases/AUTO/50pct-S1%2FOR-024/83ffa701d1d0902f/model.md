##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. Musician/band demand: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j\in J$
2. Warehouse activation: $\sum_{j\in J} x_{ij} \leq M y_i,\quad \forall i\in I$
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

Where $M = \sum_{j\in J} d_j = 1083 + 776 + 16214 = 18073$.

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

- Transportation costs $c_{ij}$:

|         | C1      | C2      | C3      |
|---------|---------|---------|---------|
| S1      | 1506.22 | 70.9    | 8.44    |
| S2      | 1732.65 | 1780.72 | 567.44  |
| S3      | 115.66  | 100.76  | 64.68   |

##### Full Model

\[
\begin{align*}
\min\quad & 1506.22\,x_{S1,C1} + 70.9\,x_{S1,C2} + 8.44\,x_{S1,C3} \\
          & + 1732.65\,x_{S2,C1} + 1780.72\,x_{S2,C2} + 567.44\,x_{S2,C3} \\
          & + 115.66\,x_{S3,C1} + 100.76\,x_{S3,C2} + 64.68\,x_{S3,C3} \\
          & + 102.33\,y_{S1} + 94.92\,y_{S2} + 91.83\,y_{S3} \\
\text{s.t.}\quad
    & x_{S1,C1} + x_{S2,C1} + x_{S3,C1} = 1083 \\
    & x_{S1,C2} + x_{S2,C2} + x_{S3,C2} = 776 \\
    & x_{S1,C3} + x_{S2,C3} + x_{S3,C3} = 16214 \\
    & x_{S1,C1} + x_{S1,C2} + x_{S1,C3} \leq 18073\,y_{S1} \\
    & x_{S2,C1} + x_{S2,C2} + x_{S2,C3} \leq 18073\,y_{S2} \\
    & x_{S3,C1} + x_{S3,C2} + x_{S3,C3} \leq 18073\,y_{S3} \\
    & x_{ij} \geq 0,\quad \forall i\in I,\,j\in J \\
    & y_i \in \{0,1\},\quad \forall i\in I
\end{align*}
\]

All parameters and indices are as retrieved from the CSV files.