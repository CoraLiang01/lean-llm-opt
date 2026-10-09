##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier (facility) $i \in I$ to branch (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier (facility) $i$ is activated (binary).

##### Parameters

- $I = \{S1, S2, S3, S4, S5\}$ (set of suppliers/facilities)
- $J = \{C1, C2, C3, C4, C5\}$ (set of branches/customers)

- Demand:
  - $d_{C1} = 143$
  - $d_{C2} = 6$
  - $d_{C3} = 10$
  - $d_{C4} = 25$
  - $d_{C5} = 3$

- Fixed opening costs:
  - $f_{S1} = 97.65$
  - $f_{S2} = 99.76$
  - $f_{S3} = 100.76$
  - $f_{S4} = 105.32$
  - $f_{S5} = 98.88$

- Transportation costs $c_{ij}$ (per unit from facility $i$ to customer $j$):

|        | C1      | C2      | C3    | C4      | C5     |
|--------|---------|---------|-------|---------|--------|
| S1     | 150.74  | 0.02    | 49.13 | 2080.15 | 426.4  |
| S2     | 233.05  | 97.73   | 49.84 | 1982.39 | 23.96  |
| S3     | 55.68   | 935.61  | 4.03  | 73.09   | 525.32 |
| S4     | 1483.82 | 1801.08 | 112.16| 816.05  | 107.01 |
| S5     | 1119.47 | 884.31  | 0.08  | 1544.95 | 543.67 |

- $M = \sum_{j \in J} d_j = 143 + 6 + 10 + 25 + 3 = 187$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   (A supplier can only ship goods if it is activated.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

###### Retrieved Information

```json
{
  "suppliers": ["S1", "S2", "S3", "S4", "S5"],
  "customers": ["C1", "C2", "C3", "C4", "C5"],
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
    "S1": {"C1": 150.74, "C2": 0.02, "C3": 49.13, "C4": 2080.15, "C5": 426.4},
    "S2": {"C1": 233.05, "C2": 97.73, "C3": 49.84, "C4": 1982.39, "C5": 23.96},
    "S3": {"C1": 55.68, "C2": 935.61, "C3": 4.03, "C4": 73.09, "C5": 525.32},
    "S4": {"C1": 1483.82, "C2": 1801.08, "C3": 112.16, "C4": 816.05, "C5": 107.01},
    "S5": {"C1": 1119.47, "C2": 884.31, "C3": 0.08, "C4": 1544.95, "C5": 543.67}
  },
  "M": 187
}
```