##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is activated (binary).

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i\in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Warehouse activation:**  
   \[
   \sum_{j\in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j\in J} d_j = 18073$ is a valid upper bound on total shipments from any warehouse.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Sets and Parameters

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

- Transportation costs $c_{ij}$:

|        | C1      | C2      | C3      |
|--------|---------|---------|---------|
| S1     | 1506.22 | 70.9    | 8.44    |
| S2     | 1732.65 | 1780.72 | 567.44  |
| S3     | 115.66  | 100.76  | 64.68   |

- $M = 1083 + 776 + 16214 = 18073$

##### Complete Mathematical Model

\[
\begin{align*}
\min\ & \sum_{i \in \{S1, S2, S3\}} \sum_{j \in \{C1, C2, C3\}} c_{ij} x_{ij} + \sum_{i \in \{S1, S2, S3\}} f_i y_i \\
\text{s.t.}\quad
& x_{S1,C1} + x_{S2,C1} + x_{S3,C1} = 1083 \\
& x_{S1,C2} + x_{S2,C2} + x_{S3,C2} = 776 \\
& x_{S1,C3} + x_{S2,C3} + x_{S3,C3} = 16214 \\
& x_{S1,C1} + x_{S1,C2} + x_{S1,C3} \leq 18073\, y_{S1} \\
& x_{S2,C1} + x_{S2,C2} + x_{S2,C3} \leq 18073\, y_{S2} \\
& x_{S3,C1} + x_{S3,C2} + x_{S3,C3} \leq 18073\, y_{S3} \\
& x_{ij} \geq 0,\quad \forall i \in \{S1, S2, S3\},\ j \in \{C1, C2, C3\} \\
& y_i \in \{0,1\},\quad \forall i \in \{S1, S2, S3\}
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