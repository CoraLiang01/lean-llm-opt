##### Objective Function:

$\quad \min \left( \sum_{i=1}^5 f_i y_i + \sum_{i=1}^5 \sum_{j=1}^5 c_{ij} x_{ij} \right)$

where:
- $f_i$ is the fixed cost of opening supplier $S_i$
- $y_i$ is a binary variable indicating if supplier $S_i$ is open ($y_i \in \{0,1\}$)
- $c_{ij}$ is the transportation cost per unit from supplier $S_i$ to branch $C_j$
- $x_{ij}$ is the quantity supplied from $S_i$ to $C_j$

##### Constraints:

1. **Demand Satisfaction (for each branch):**

$\sum_{i=1}^5 x_{ij} = d_j \quad \forall j \in \{1,2,3,4,5\}$

where $d_j$ is the demand of branch $C_j$.

2. **Supplier Activation (linking $x_{ij}$ to $y_i$):**

$x_{ij} \leq d_j y_i \quad \forall i \in \{1,2,3,4,5\},\ \forall j \in \{1,2,3,4,5\}$

3. **Variable Domains:**

$y_i \in \{0,1\} \quad \forall i \in \{1,2,3,4,5\}$

$x_{ij} \geq 0 \quad \forall i \in \{1,2,3,4,5\},\ \forall j \in \{1,2,3,4,5\}$

##### Retrieved Information

```json
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
  "demands": {
    "C1": 143,
    "C2": 6,
    "C3": 10,
    "C4": 25,
    "C5": 3
  },
  "transportation_costs": {
    "S1": {"C1": 150.74, "C2": 0.02, "C3": 49.13, "C4": 2080.15, "C5": 426.4},
    "S2": {"C1": 233.05, "C2": 97.73, "C3": 49.84, "C4": 1982.39, "C5": 23.96},
    "S3": {"C1": 55.68, "C2": 935.61, "C3": 4.03, "C4": 73.09, "C5": 525.32},
    "S4": {"C1": 1483.82, "C2": 1801.08, "C3": 112.16, "C4": 816.05, "C5": 107.01},
    "S5": {"C1": 1119.47, "C2": 884.31, "C3": 0.08, "C4": 1544.95, "C5": 543.67}
  }
}
```

##### Full Model with Parameters

Let $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ and $j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$.

**Objective:**

$\min \Bigg[ \ $
$\quad 97.65\, y_{\text{S1}} + 99.76\, y_{\text{S2}} + 100.76\, y_{\text{S3}} + 105.32\, y_{\text{S4}} + 98.88\, y_{\text{S5}}$
$+ \left(150.74\, x_{\text{S1},\text{C1}} + 0.02\, x_{\text{S1},\text{C2}} + 49.13\, x_{\text{S1},\text{C3}} + 2080.15\, x_{\text{S1},\text{C4}} + 426.4\, x_{\text{S1},\text{C5}}\right)$
$+ \left(233.05\, x_{\text{S2},\text{C1}} + 97.73\, x_{\text{S2},\text{C2}} + 49.84\, x_{\text{S2},\text{C3}} + 1982.39\, x_{\text{S2},\text{C4}} + 23.96\, x_{\text{S2},\text{C5}}\right)$
$+ \left(55.68\, x_{\text{S3},\text{C1}} + 935.61\, x_{\text{S3},\text{C2}} + 4.03\, x_{\text{S3},\text{C3}} + 73.09\, x_{\text{S3},\text{C4}} + 525.32\, x_{\text{S3},\text{C5}}\right)$
$+ \left(1483.82\, x_{\text{S4},\text{C1}} + 1801.08\, x_{\text{S4},\text{C2}} + 112.16\, x_{\text{S4},\text{C3}} + 816.05\, x_{\text{S4},\text{C4}} + 107.01\, x_{\text{S4},\text{C5}}\right)$
$+ \left(1119.47\, x_{\text{S5},\text{C1}} + 884.31\, x_{\text{S5},\text{C2}} + 0.08\, x_{\text{S5},\text{C3}} + 1544.95\, x_{\text{S5},\text{C4}} + 543.67\, x_{\text{S5},\text{C5}}\right)$
$\ \Bigg]$

**Subject to:**

- Demand satisfaction for each branch:
  - $x_{\text{S1},j} + x_{\text{S2},j} + x_{\text{S3},j} + x_{\text{S4},j} + x_{\text{S5},j} = d_j$ for $j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$

    - $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} + x_{\text{S4},\text{C1}} + x_{\text{S5},\text{C1}} = 143$
    - $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} + x_{\text{S4},\text{C2}} + x_{\text{S5},\text{C2}} = 6$
    - $x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} + x_{\text{S4},\text{C3}} + x_{\text{S5},\text{C3}} = 10$
    - $x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + x_{\text{S3},\text{C4}} + x_{\text{S4},\text{C4}} + x_{\text{S5},\text{C4}} = 25$
    - $x_{\text{S1},\text{C5}} + x_{\text{S2},\text{C5}} + x_{\text{S3},\text{C5}} + x_{\text{S4},\text{C5}} + x_{\text{S5},\text{C5}} = 3$

- Linking constraints:
  - $x_{ij} \leq d_j y_i$ for all $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$, $j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$

- Variable domains:
  - $y_i \in \{0,1\}$ for all $i$
  - $x_{ij} \geq 0$ for all $i, j$