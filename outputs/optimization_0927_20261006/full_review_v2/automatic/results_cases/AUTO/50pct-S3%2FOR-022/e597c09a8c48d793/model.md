##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. **Branch demand:**  
   \[
   \sum_{i\in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation:**  
   \[
   \sum_{j\in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j\in J} d_j = 187$.
3. **Domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Sets and Parameters

- $I = \{S1, S2, S3, S4, S5\}$ (suppliers)
- $J = \{C1, C2, C3, C4, C5\}$ (branches)

- Demands $d_j$:
  - $d_{C1} = 143$
  - $d_{C2} = 6$
  - $d_{C3} = 10$
  - $d_{C4} = 25$
  - $d_{C5} = 3$

- Fixed opening costs $f_i$:
  - $f_{S1} = 97.65$
  - $f_{S2} = 99.76$
  - $f_{S3} = 100.76$
  - $f_{S4} = 105.32$
  - $f_{S5} = 98.88$

- Transportation costs $c_{ij}$:

|        | C1      | C2     | C3    | C4      | C5     |
|--------|---------|--------|-------|---------|--------|
| S1     | 150.74  | 0.02   | 49.13 | 2080.15 | 426.4  |
| S2     | 233.05  | 97.73  | 49.84 | 1982.39 | 23.96  |
| S3     | 55.68   | 935.61 | 4.03  | 73.09   | 525.32 |
| S4     | 1483.82 | 1801.08| 112.16| 816.05  | 107.01 |
| S5     | 1119.47 | 884.31 | 0.08  | 1544.95 | 543.67 |

- $M = 143 + 6 + 10 + 25 + 3 = 187$

##### Full Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 187\, y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

###### Retrieved Information

{
  "suppliers": [
    "S1",
    "S2",
    "S3",
    "S4",
    "S5"
  ],
  "branches": [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5"
  ],
  "demand": {
    "C1": 143,
    "C2": 6,
    "C3": 10,
    "C4": 25,
    "C5": 3
  },
  "fixed_cost": {
    "S1": 97.65,
    "S2": 99.76,
    "S3": 100.76,
    "S4": 105.32,
    "S5": 98.88
  },
  "cost": {
    "S1": {
      "C1": 150.74,
      "C2": 0.02,
      "C3": 49.13,
      "C4": 2080.15,
      "C5": 426.4
    },
    "S2": {
      "C1": 233.05,
      "C2": 97.73,
      "C3": 49.84,
      "C4": 1982.39,
      "C5": 23.96
    },
    "S3": {
      "C1": 55.68,
      "C2": 935.61,
      "C3": 4.03,
      "C4": 73.09,
      "C5": 525.32
    },
    "S4": {
      "C1": 1483.82,
      "C2": 1801.08,
      "C3": 112.16,
      "C4": 816.05,
      "C5": 107.01
    },
    "S5": {
      "C1": 1119.47,
      "C2": 884.31,
      "C3": 0.08,
      "C4": 1544.95,
      "C5": 543.67
    }
  },
  "M": 187
}