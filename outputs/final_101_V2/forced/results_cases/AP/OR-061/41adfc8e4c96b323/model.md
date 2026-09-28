##### Objective Function:

$\quad \min \left( \sum_{i=1}^5 f_i y_i + \sum_{i=1}^5 \sum_{j=1}^5 c_{ij} x_{ij} \right)$

where:
- $f_i$ is the fixed cost of opening supplier $S_i$
- $c_{ij}$ is the transportation cost per unit from supplier $S_i$ to branch $C_j$
- $y_i$ is a binary variable indicating if supplier $S_i$ is open ($y_i \in \{0,1\}$)
- $x_{ij}$ is the quantity supplied from $S_i$ to $C_j$ ($x_{ij} \geq 0$)

##### Constraints

###### 1. Demand Satisfaction (for each branch):

$\sum_{i=1}^5 x_{ij} = d_j \quad \forall j \in \{1,2,3,4,5\}$

where $d_j$ is the demand of branch $C_j$.

###### 2. Supplier Activation (linking $x_{ij}$ to $y_i$):

$x_{ij} \leq d_j y_i \quad \forall i \in \{1,2,3,4,5\},\ \forall j \in \{1,2,3,4,5\}$

###### 3. Variable Domains:

$y_i \in \{0,1\} \quad \forall i \in \{1,2,3,4,5\}$

$x_{ij} \geq 0 \quad \forall i \in \{1,2,3,4,5\},\ \forall j \in \{1,2,3,4,5\}$

---

##### Retrieved Information

```json
{
  "branches": [
    {"id": "C1", "demand": 143},
    {"id": "C2", "demand": 6},
    {"id": "C3", "demand": 10},
    {"id": "C4", "demand": 25},
    {"id": "C5", "demand": 3}
  ],
  "suppliers": [
    {"id": "S1", "fixed_cost": 97.65},
    {"id": "S2", "fixed_cost": 99.76},
    {"id": "S3", "fixed_cost": 100.76},
    {"id": "S4", "fixed_cost": 105.32},
    {"id": "S5", "fixed_cost": 98.88}
  ],
  "transportation_costs": {
    "S1": {"C1": 150.74, "C2": 0.02, "C3": 49.13, "C4": 2080.15, "C5": 426.4},
    "S2": {"C1": 233.05, "C2": 97.73, "C3": 49.84, "C4": 1982.39, "C5": 23.96},
    "S3": {"C1": 55.68, "C2": 935.61, "C3": 4.03, "C4": 73.09, "C5": 525.32},
    "S4": {"C1": 1483.82, "C2": 1801.08, "C3": 112.16, "C4": 816.05, "C5": 107.01},
    "S5": {"C1": 1119.47, "C2": 884.31, "C3": 0.08, "C4": 1544.95, "C5": 543.67}
  }
}
```

- Branches: $C_1, C_2, C_3, C_4, C_5$
- Suppliers: $S_1, S_2, S_3, S_4, S_5$
- Demands: $d_1 = 143$, $d_2 = 6$, $d_3 = 10$, $d_4 = 25$, $d_5 = 3$
- Fixed costs: $f_1 = 97.65$, $f_2 = 99.76$, $f_3 = 100.76$, $f_4 = 105.32$, $f_5 = 98.88$
- Transportation costs matrix $[c_{ij}]$:

|        | C1      | C2      | C3     | C4      | C5     |
|--------|---------|---------|--------|---------|--------|
| S1     | 150.74  | 0.02    | 49.13  | 2080.15 | 426.40 |
| S2     | 233.05  | 97.73   | 49.84  | 1982.39 | 23.96  |
| S3     | 55.68   | 935.61  | 4.03   | 73.09   | 525.32 |
| S4     | 1483.82 | 1801.08 | 112.16 | 816.05  | 107.01 |
| S5     | 1119.47 | 884.31  | 0.08   | 1544.95 | 543.67 |

##### Decision Variables

- $y_i$: Binary, $1$ if supplier $S_i$ is open, $0$ otherwise.
- $x_{ij}$: Quantity supplied from $S_i$ to $C_j$, continuous and non-negative.

##### Complete Mathematical Model

$\boxed{
\begin{align*}
\min\ & \sum_{i=1}^5 f_i y_i + \sum_{i=1}^5 \sum_{j=1}^5 c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i=1}^5 x_{ij} = d_j \quad \forall j \in \{1,2,3,4,5\} \\
& x_{ij} \leq d_j y_i \quad \forall i \in \{1,2,3,4,5\},\ \forall j \in \{1,2,3,4,5\} \\
& y_i \in \{0,1\} \quad \forall i \in \{1,2,3,4,5\} \\
& x_{ij} \geq 0 \quad \forall i \in \{1,2,3,4,5\},\ \forall j \in \{1,2,3,4,5\}
\end{align*}
}$