##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. Demand satisfaction for each musician/band:
   \[
   \sum_{i\in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Warehouse activation: shipments only from active warehouses:
   \[
   \sum_{j\in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j\in J} d_j = 18073$.
3. Variable domains:
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Sets and Parameters

- Warehouses: $I = \{S1, S2, S3\}$
- Musicians/Bands: $J = \{C1, C2, C3\}$

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

- $M = 1083 + 776 + 16214 = 18073$

##### Full Model

\[
\begin{align*}
\min\ & \big[1506.22\,x_{S1,C1} + 70.9\,x_{S1,C2} + 8.44\,x_{S1,C3} \\
&\quad + 1732.65\,x_{S2,C1} + 1780.72\,x_{S2,C2} + 567.44\,x_{S2,C3} \\
&\quad + 115.66\,x_{S3,C1} + 100.76\,x_{S3,C2} + 64.68\,x_{S3,C3}\big] \\
&\quad + 102.33\,y_{S1} + 94.92\,y_{S2} + 91.83\,y_{S3} \\
\text{s.t.}\quad
& x_{S1,C1} + x_{S2,C1} + x_{S3,C1} = 1083 \\
& x_{S1,C2} + x_{S2,C2} + x_{S3,C2} = 776 \\
& x_{S1,C3} + x_{S2,C3} + x_{S3,C3} = 16214 \\
& x_{S1,C1} + x_{S1,C2} + x_{S1,C3} \leq 18073\,y_{S1} \\
& x_{S2,C1} + x_{S2,C2} + x_{S2,C3} \leq 18073\,y_{S2} \\
& x_{S3,C1} + x_{S3,C2} + x_{S3,C3} \leq 18073\,y_{S3} \\
& x_{ij} \geq 0,\quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\},\quad \forall i \in I
\end{align*}
\]

###### Retrieved Information

{
  "warehouses": ["S1", "S2", "S3"],
  "musicians_bands": ["C1", "C2", "C3"],
  "demand": {
    "C1": 1083,
    "C2": 776,
    "C3": 16214
  },
  "fixed_cost": {
    "S1": 102.33,
    "S2": 94.92,
    "S3": 91.83
  },
  "cost": {
    "S1": {"C1": 1506.22, "C2": 70.9, "C3": 8.44},
    "S2": {"C1": 1732.65, "C2": 1780.72, "C3": 567.44},
    "S3": {"C1": 115.66, "C2": 100.76, "C3": 64.68}
  },
  "M": 18073
}