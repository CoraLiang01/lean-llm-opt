##### Sets

- Suppliers: $I = \{S1, S2, S3, S4, S5\}$
- Branches: $J = \{C1, C2, C3, C4, C5\}$

##### Parameters

- Fixed costs for suppliers:
  - $f_{S1} = 97.65$
  - $f_{S2} = 99.76$
  - $f_{S3} = 100.76$
  - $f_{S4} = 105.32$
  - $f_{S5} = 98.88$

- Transportation costs per unit from supplier $i$ to branch $j$ ($c_{ij}$):

|        | C1      | C2      | C3      | C4      | C5      |
|--------|---------|---------|---------|---------|---------|
| S1     | 150.74  | 0.02    | 49.13   | 2080.15 | 426.40  |
| S2     | 233.05  | 97.73   | 49.84   | 1982.39 | 23.96   |
| S3     | 55.68   | 935.61  | 4.03    | 73.09   | 525.32  |
| S4     | 1483.82 | 1801.08 | 112.16  | 816.05  | 107.01  |
| S5     | 1119.47 | 884.31  | 0.08    | 1544.95 | 543.67  |

- Demand for each branch:
  - $d_{C1} = 143$
  - $d_{C2} = 6$
  - $d_{C3} = 10$
  - $d_{C4} = 25$
  - $d_{C5} = 3$

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise, for $i \in I$
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to branch $j$, for $i \in I$, $j \in J$

##### Objective Function

$$
\min \left( \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \right)
$$

##### Constraints

1. **Demand Satisfaction:**  
   For each branch $j \in J$,
   $$
   \sum_{i \in I} x_{ij} = d_j
   $$

2. **Supplier Activation:**  
   For each supplier $i \in I$ and branch $j \in J$,
   $$
   x_{ij} \leq d_j y_i
   $$
   (A supplier can only supply to a branch if it is open; $d_j$ is an upper bound.)

3. **Variable Domains:**  
   $$
   y_i \in \{0,1\} \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   $$

##### Retrieved Information

{
  "suppliers": ["S1", "S2", "S3", "S4", "S5"],
  "branches": ["C1", "C2", "C3", "C4", "C5"],
  "fixed_costs": {
    "S1": 97.65,
    "S2": 99.76,
    "S3": 100.76,
    "S4": 105.32,
    "S5": 98.88
  },
  "transportation_costs": {
    "S1": {"C1": 150.74, "C2": 0.02, "C3": 49.13, "C4": 2080.15, "C5": 426.40},
    "S2": {"C1": 233.05, "C2": 97.73, "C3": 49.84, "C4": 1982.39, "C5": 23.96},
    "S3": {"C1": 55.68, "C2": 935.61, "C3": 4.03, "C4": 73.09, "C5": 525.32},
    "S4": {"C1": 1483.82, "C2": 1801.08, "C3": 112.16, "C4": 816.05, "C5": 107.01},
    "S5": {"C1": 1119.47, "C2": 884.31, "C3": 0.08, "C4": 1544.95, "C5": 543.67}
  },
  "demand": {
    "C1": 143,
    "C2": 6,
    "C3": 10,
    "C4": 25,
    "C5": 3
  }
}