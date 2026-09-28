##### Sets and Indices

- Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ be the set of suppliers, indexed by $i$.
- Let $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$ be the set of branches (customers), indexed by $j$.

##### Parameters

- $f_i$: Fixed cost of opening supplier $i$.
    - $f_{\text{S1}} = 97.65$
    - $f_{\text{S2}} = 99.76$
    - $f_{\text{S3}} = 100.76$
    - $f_{\text{S4}} = 105.32$
    - $f_{\text{S5}} = 98.88$
- $c_{ij}$: Transportation cost per unit from supplier $i$ to branch $j$.

    |        | C1      | C2      | C3     | C4      | C5      |
    |--------|---------|---------|--------|---------|---------|
    | S1     | 150.74  | 0.02    | 49.13  | 2080.15 | 426.40  |
    | S2     | 233.05  | 97.73   | 49.84  | 1982.39 | 23.96   |
    | S3     | 55.68   | 935.61  | 4.03   | 73.09   | 525.32  |
    | S4     | 1483.82 | 1801.08 | 112.16 | 816.05  | 107.01  |
    | S5     | 1119.47 | 884.31  | 0.08   | 1544.95 | 543.67  |

- $d_j$: Demand at branch $j$.
    - $d_{\text{C1}} = 143$
    - $d_{\text{C2}} = 6$
    - $d_{\text{C3}} = 10$
    - $d_{\text{C4}} = 25$
    - $d_{\text{C5}} = 3$

##### Decision Variables

- $y_i \in \{0,1\}$: $1$ if supplier $i$ is open, $0$ otherwise.
- $x_{ij} \geq 0$: Quantity supplied from supplier $i$ to branch $j$.

##### Mathematical Model

Minimize total cost:
$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:

1. **Demand satisfaction at each branch:**
   $$
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   $$

2. **Supply only from open suppliers:**
   $$
   x_{ij} \leq d_j y_i, \quad \forall i \in I, \forall j \in J
   $$

3. **Variable domains:**
   $$
   y_i \in \{0,1\}, \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0, \quad \forall i \in I, \forall j \in J
   $$

##### Data Used

- Demand (demand.csv):

    | customer | demand |
    |----------|--------|
    | C1       | 143    |
    | C2       | 6      |
    | C3       | 10     |
    | C4       | 25     |
    | C5       | 3      |

- Fixed costs (fixed_cost.csv):

    | Supplier | fixed_costs |
    |----------|-------------|
    | S1       | 97.65       |
    | S2       | 99.76       |
    | S3       | 100.76      |
    | S4       | 105.32      |
    | S5       | 98.88       |

- Transportation costs (transportation_costs.csv):

    |        | C1      | C2      | C3     | C4      | C5      |
    |--------|---------|---------|--------|---------|---------|
    | S1     | 150.74  | 0.02    | 49.13  | 2080.15 | 426.40  |
    | S2     | 233.05  | 97.73   | 49.84  | 1982.39 | 23.96   |
    | S3     | 55.68   | 935.61  | 4.03   | 73.09   | 525.32  |
    | S4     | 1483.82 | 1801.08 | 112.16 | 816.05  | 107.01  |
    | S5     | 1119.47 | 884.31  | 0.08   | 1544.95 | 543.67  |