Let:
- $S = \{S1, S2, \ldots, S12\}$ be the set of suppliers (indexed by $i$).
- $C = \{C1, C2, \ldots, C12\}$ be the set of supermarkets/customers (indexed by $j$).
- $f_i$ be the fixed cost of opening supplier $i$.
- $t_{ij}$ be the transportation cost per unit from supplier $i$ to customer $j$.
- $d_j$ be the demand of customer $j$.
- $y_i \in \{0,1\}$ indicates if supplier $i$ is open.
- $x_{ij} \geq 0$ is the quantity supplied from supplier $i$ to customer $j$.

Parameters (from the data):

- Fixed costs $f_i$:

| Supplier | Fixed Cost |
|----------|------------|
| S1       | 98.88      |
| S2       | 99.73      |
| S3       | 94.01      |
| S4       | 93.77      |
| S5       | 107.59     |
| S6       | 112.65     |
| S7       | 97.05      |
| S8       | 103        |
| S9       | 90.45      |
| S10      | 96.73      |
| S11      | 96.43      |
| S12      | 112.19     |

- Demands $d_j$:

| Customer | Demand |
|----------|--------|
| C1       | 1097   |
| C2       | 61     |
| C3       | 11     |
| C4       | 7      |
| C5       | 82     |
| C6       | 37     |
| C7       | 483    |
| C8       | 582    |
| C9       | 223    |
| C10      | 89     |
| C11      | 60     |
| C12      | 55     |

- Transportation costs $t_{ij}$ (partial table shown for illustration; all values from the data must be used):

|        | C1     | C2     | C3     | C4     | C5     | C6     | C7     | C8     | C9     | C10    | C11    | C12    |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| S1     | 284.11 | 53.78  | 10.62  | 111.27 | 158.5  | 8.79   | 53.79  | 8.84   | 1911.43| 8.87   | 1129.47| 185.53 |
| S2     | 7.19   | 1031.96| 90.94  | 276.97 | 0.45   | 0.2    | 49.14  | 1.05   | 2079.54| 1.45   | 49.14  | 0.05   |
| ...    | ...    | ...    | ...    | ...    | ...    | ...    | ...    | ...    | ...    | ...    | ...    | ...    |

Model:

Minimize total cost:
$$
\min \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} t_{ij} x_{ij}
$$

Subject to:

1. Demand satisfaction for each customer:
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

All parameters ($f_i$, $t_{ij}$, $d_j$) are as given in the retrieved data above, with all indices and coefficients preserved.

This model determines which suppliers to open and how much each should supply to each supermarket to minimize the sum of fixed and transportation costs, while meeting all supermarket demands.