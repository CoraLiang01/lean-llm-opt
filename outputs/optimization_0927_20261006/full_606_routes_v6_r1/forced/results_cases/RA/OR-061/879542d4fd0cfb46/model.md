Let:
- $S = \{S1, S2, S3, S4, S5\}$ be the set of suppliers.
- $C = \{C1, C2, C3, C4, C5\}$ be the set of branches (customers).
- $y_i \in \{0,1\}$ indicates if supplier $i$ is open.
- $x_{ij} \geq 0$ is the quantity supplied from supplier $i$ to branch $j$.

Parameters (from data):

- Fixed costs:
  - $f_{S1} = 97.65$
  - $f_{S2} = 99.76$
  - $f_{S3} = 100.76$
  - $f_{S4} = 105.32$
  - $f_{S5} = 98.88$

- Demands:
  - $d_{C1} = 143$
  - $d_{C2} = 6$
  - $d_{C3} = 10$
  - $d_{C4} = 25$
  - $d_{C5} = 3$

- Transportation costs $t_{ij}$:

|        | C1      | C2      | C3    | C4      | C5      |
|--------|---------|---------|-------|---------|---------|
| S1     | 150.74  | 0.02    | 49.13 | 2080.15 | 426.40  |
| S2     | 233.05  | 97.73   | 49.84 | 1982.39 | 23.96   |
| S3     | 55.68   | 935.61  | 4.03  | 73.09   | 525.32  |
| S4     | 1483.82 | 1801.08 | 112.16| 816.05  | 107.01  |
| S5     | 1119.47 | 884.31  | 0.08  | 1544.95 | 543.67  |

Model:

Minimize total cost:
$$
\min \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} t_{ij} x_{ij}
$$

Subject to:

1. Demand satisfaction for each branch:
$$
\sum_{i \in S} x_{ij} = d_j \quad \forall j \in C
$$

2. Supply only from open suppliers:
$$
x_{ij} \leq d_j y_i \quad \forall i \in S, \forall j \in C
$$

3. Variable domains:
$$
y_i \in \{0,1\} \quad \forall i \in S \\
x_{ij} \geq 0 \quad \forall i \in S, \forall j \in C
$$

Where:
- $f_i$ is the fixed cost for supplier $i$ (see above).
- $t_{ij}$ is the transportation cost per unit from supplier $i$ to branch $j$ (see table above).
- $d_j$ is the demand for branch $j$ (see above).

All identifiers and coefficients are as retrieved and preserved in source order.