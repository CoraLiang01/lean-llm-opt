##### Objective Function:

$\quad \min \sum_{i=1}^6 f_i y_i + \sum_{i=1}^6 \sum_{j=1}^6 c_{ij} x_{ij}$

where:
- $f_i$ is the fixed cost of opening supplier $S_i$
- $c_{ij}$ is the per-unit transportation cost from supplier $S_i$ to customer $C_j$
- $y_i \in \{0,1\}$ indicates if supplier $S_i$ is open
- $x_{ij} \geq 0$ is the quantity shipped from supplier $S_i$ to customer $C_j$

##### Constraints

###### 1. Demand Satisfaction:

$\sum_{i=1}^6 x_{ij} = d_j \quad \forall j \in \{1,2,3,4,5,6\}$

where $d_j$ is the demand at customer $C_j$.

###### 2. Supplier Activation:

$\sum_{j=1}^6 x_{ij} \leq M \cdot y_i \quad \forall i \in \{1,2,3,4,5,6\}$

where $M$ is a sufficiently large constant (e.g., $M = \sum_j d_j$).

###### 3. Variable Domains:

$y_i \in \{0,1\} \quad \forall i \in \{1,2,3,4,5,6\}$

$x_{ij} \geq 0 \quad \forall i \in \{1,2,3,4,5,6\},\; j \in \{1,2,3,4,5,6\}$

---

##### Retrieved Information

{
  "suppliers": [
    {"id": "S1", "fixed_cost": 98.88},
    {"id": "S2", "fixed_cost": 99.73},
    {"id": "S3", "fixed_cost": 94.01},
    {"id": "S4", "fixed_cost": 93.77},
    {"id": "S5", "fixed_cost": 107.59},
    {"id": "S6", "fixed_cost": 112.65}
  ],
  "transportation_cost": {
    "S1": {"C1": 0.08, "C2": 52.33, "C3": 73.57, "C4": 1237.33, "C5": 0.07, "C6": 112.16},
    "S2": {"C1": 46.02, "C2": 175.23, "C3": 2026.83, "C4": 299.89, "C5": 966.53, "C6": 1590.42},
    "S3": {"C1": 1031.74, "C2": 78.13, "C3": 99.02, "C4": 277.07, "C5": 884.45, "C6": 1800.86},
    "S4": {"C1": 868.75, "C2": 94.2, "C3": 1776.34, "C4": 285.48, "C5": 868.85, "C6": 86.55},
    "S5": {"C1": 1577, "C2": 760.15, "C3": 2090.19, "C4": 43.2, "C5": 1577.12, "C6": 1095.17},
    "S6": {"C1": 49.14, "C2": 4.33, "C3": 2079.57, "C4": 277.04, "C5": 1032.01, "C6": 1543.49}
  },
  "demand": {
    "C1": 216,
    "C2": 216,
    "C3": 216,
    "C4": 144,
    "C5": 144,
    "C6": 144
  }
}

- Suppliers: S1, S2, S3, S4, S5, S6
- Customers: C1, C2, C3, C4, C5, C6
- Fixed costs: $f_1 = 98.88$, $f_2 = 99.73$, $f_3 = 94.01$, $f_4 = 93.77$, $f_5 = 107.59$, $f_6 = 112.65$
- Transportation costs $c_{ij}$ as above
- Demands: $d_1 = 216$, $d_2 = 216$, $d_3 = 216$, $d_4 = 144$, $d_5 = 144$, $d_6 = 144$