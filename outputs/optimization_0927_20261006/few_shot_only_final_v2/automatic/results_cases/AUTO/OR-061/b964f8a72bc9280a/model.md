##### Parameters

- Suppliers $I = \{S1, S2, S3, S4, S5\}$
- Branches $J = \{C1, C2, C3, C4, C5\}$

- Demands:
  - $d_{C1} = 143$
  - $d_{C2} = 6$
  - $d_{C3} = 10$
  - $d_{C4} = 25$
  - $d_{C5} = 3$

- Fixed costs:
  - $f_{S1} = 97.65$
  - $f_{S2} = 99.76$
  - $f_{S3} = 100.76$
  - $f_{S4} = 105.32$
  - $f_{S5} = 98.88$

- Transportation costs $c_{ij}$:

|        | C1      | C2      | C3     | C4      | C5      |
|--------|---------|---------|--------|---------|---------|
| S1     | 150.74  | 0.02    | 49.13  | 2080.15 | 426.40  |
| S2     | 233.05  | 97.73   | 49.84  | 1982.39 | 23.96   |
| S3     | 55.68   | 935.61  | 4.03   | 73.09   | 525.32  |
| S4     | 1483.82 | 1801.08 | 112.16 | 816.05  | 107.01  |
| S5     | 1119.47 | 884.31  | 0.08   | 1544.95 | 543.67  |

- $M = \sum_{j \in J} d_j = 143 + 6 + 10 + 25 + 3 = 187$

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous)
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise (binary)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Branch demand satisfaction** (each branch receives exactly its demand):

   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation** (inactive suppliers cannot ship):

   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains**:

   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Model Data

- $I = \{S1, S2, S3, S4, S5\}$
- $J = \{C1, C2, C3, C4, C5\}$
- $d_{C1} = 143$, $d_{C2} = 6$, $d_{C3} = 10$, $d_{C4} = 25$, $d_{C5} = 3$
- $f_{S1} = 97.65$, $f_{S2} = 99.76$, $f_{S3} = 100.76$, $f_{S4} = 105.32$, $f_{S5} = 98.88$
- $c_{ij}$ as in the table above
- $M = 187$