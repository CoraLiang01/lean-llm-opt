##### Objective Function:

$\quad \min \left( \sum_{i=1}^3 f_i y_i + \sum_{i=1}^3 \sum_{j=1}^3 c_{ij} x_{ij} \right)$

where:
- $f_i$ is the fixed cost of opening warehouse $S_i$
- $y_i$ is a binary variable indicating if warehouse $S_i$ is open ($y_i \in \{0,1\}$)
- $c_{ij}$ is the transportation cost per unit from warehouse $S_i$ to customer $C_j$
- $x_{ij}$ is the quantity shipped from warehouse $S_i$ to customer $C_j$

##### Constraints:

1. **Demand Satisfaction:**

$\sum_{i=1}^3 x_{ij} = d_j \quad \forall j \in \{1,2,3\}$

where $d_j$ is the demand of customer $C_j$.

2. **Warehouse Activation:**

$x_{ij} \leq d_j y_i \quad \forall i \in \{1,2,3\},\ \forall j \in \{1,2,3\}$

(If warehouse $S_i$ is not open, it cannot supply any customer.)

3. **Variable Domains:**

$y_i \in \{0,1\} \quad \forall i \in \{1,2,3\}$

$x_{ij} \geq 0 \quad \forall i \in \{1,2,3\},\ \forall j \in \{1,2,3\}$

##### Retrieved Information

{
  "warehouses": [
    "S1",
    "S2",
    "S3"
  ],
  "customers": [
    "C1",
    "C2",
    "C3"
  ],
  "fixed_costs": {
    "S1": 102.33,
    "S2": 94.92,
    "S3": 91.83
  },
  "demands": {
    "C1": 1083,
    "C2": 776,
    "C3": 16214
  },
  "transportation_costs": {
    "S1": {
      "C1": 1506.22,
      "C2": 70.90,
      "C3": 8.44
    },
    "S2": {
      "C1": 1732.65,
      "C2": 1780.72,
      "C3": 567.44
    },
    "S3": {
      "C1": 115.66,
      "C2": 100.76,
      "C3": 64.68
    }
  }
}

##### Full Parameter Matrices

- Fixed costs: $f = [102.33, 94.92, 91.83]$ for $S1$, $S2$, $S3$
- Demands: $d = [1083, 776, 16214]$ for $C1$, $C2$, $C3$
- Transportation costs $c_{ij}$:

|        | C1      | C2      | C3      |
|--------|---------|---------|---------|
| S1     | 1506.22 | 70.90   | 8.44    |
| S2     | 1732.65 | 1780.72 | 567.44  |
| S3     | 115.66  | 100.76  | 64.68   |

##### Decision Variables

- $y_i$: Binary, $1$ if warehouse $S_i$ is open, $0$ otherwise.
- $x_{ij}$: Non-negative, quantity shipped from $S_i$ to $C_j$.

##### Sets

- Warehouses: $S = \{S1, S2, S3\}$
- Customers: $C = \{C1, C2, C3\}$

##### Complete Model

$\min \left( 102.33\,y_1 + 94.92\,y_2 + 91.83\,y_3 + 1506.22\,x_{11} + 70.90\,x_{12} + 8.44\,x_{13} + 1732.65\,x_{21} + 1780.72\,x_{22} + 567.44\,x_{23} + 115.66\,x_{31} + 100.76\,x_{32} + 64.68\,x_{33} \right)$

Subject to:

$\quad x_{11} + x_{21} + x_{31} = 1083$

$\quad x_{12} + x_{22} + x_{32} = 776$

$\quad x_{13} + x_{23} + x_{33} = 16214$

$\quad x_{1j} \leq d_j y_1 \quad \forall j \in \{1,2,3\}$

$\quad x_{2j} \leq d_j y_2 \quad \forall j \in \{1,2,3\}$

$\quad x_{3j} \leq d_j y_3 \quad \forall j \in \{1,2,3\}$

$\quad y_i \in \{0,1\} \quad \forall i \in \{1,2,3\}$

$\quad x_{ij} \geq 0 \quad \forall i \in \{1,2,3\},\ \forall j \in \{1,2,3\}$