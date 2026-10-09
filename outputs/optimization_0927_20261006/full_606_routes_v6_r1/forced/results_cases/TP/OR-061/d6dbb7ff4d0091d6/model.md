##### Sets

- Suppliers: $I = \{S1, S2, S3, S4, S5\}$
- Branches (customers): $J = \{C1, C2, C3, C4, C5\}$

##### Parameters

- Demand for each branch:
  - $d_{C1} = 143$
  - $d_{C2} = 6$
  - $d_{C3} = 10$
  - $d_{C4} = 25$
  - $d_{C5} = 3$
- Fixed cost for each supplier:
  - $f_{S1} = 97.65$
  - $f_{S2} = 99.76$
  - $f_{S3} = 100.76$
  - $f_{S4} = 105.32$
  - $f_{S5} = 98.88$
- Transportation cost per unit from supplier $i$ to branch $j$ ($c_{ij}$):

|        | C1      | C2      | C3     | C4      | C5      |
|--------|---------|---------|--------|---------|---------|
| S1     | 150.74  | 0.02    | 49.13  | 2080.15 | 426.40  |
| S2     | 233.05  | 97.73   | 49.84  | 1982.39 | 23.96   |
| S3     | 55.68   | 935.61  | 4.03   | 73.09   | 525.32  |
| S4     | 1483.82 | 1801.08 | 112.16 | 816.05  | 107.01  |
| S5     | 1119.47 | 884.31  | 0.08   | 1544.95 | 543.67  |

##### Decision Variables

- $y_i \in \{0,1\}$: $1$ if supplier $i$ is operational (open), $0$ otherwise, for all $i \in I$.
- $x_{ij} \geq 0$: quantity of goods supplied from supplier $i$ to branch $j$, for all $i \in I$, $j \in J$.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:** For each branch $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]
2. **Activation constraint:** For all $i \in I$, $j \in J$,
   \[
   x_{ij} \leq M_{ij} y_i
   \]
   where $M_{ij}$ is a sufficiently large constant (e.g., $M_{ij} \geq d_j$), ensuring that supplier $i$ can only supply to branch $j$ if it is open.

3. **Variable domains:**
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data (retrieved)

- Demand:
  - C1: 143
  - C2: 6
  - C3: 10
  - C4: 25
  - C5: 3
- Fixed costs:
  - S1: 97.65
  - S2: 99.76
  - S3: 100.76
  - S4: 105.32
  - S5: 98.88
- Transportation costs:

|        | C1      | C2      | C3     | C4      | C5      |
|--------|---------|---------|--------|---------|---------|
| S1     | 150.74  | 0.02    | 49.13  | 2080.15 | 426.40  |
| S2     | 233.05  | 97.73   | 49.84  | 1982.39 | 23.96   |
| S3     | 55.68   | 935.61  | 4.03   | 73.09   | 525.32  |
| S4     | 1483.82 | 1801.08 | 112.16 | 816.05  | 107.01  |
| S5     | 1119.47 | 884.31  | 0.08   | 1544.95 | 543.67  |