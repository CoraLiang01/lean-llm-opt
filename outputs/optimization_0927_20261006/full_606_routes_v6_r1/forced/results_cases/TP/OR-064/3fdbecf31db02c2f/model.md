##### Sets

Let $I = \{S1, S2, \ldots, S24\}$ be the set of suppliers.  
Let $J = \{C1, C2, \ldots, C25\}$ be the set of supermarkets.

##### Parameters

- $f_i$: fixed cost of opening supplier $i \in I$ (from fixed_cost.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to supermarket $j$ (from transportation_costs.csv)
- $d_j$: demand of supermarket $j \in J$ (from demand.csv)

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to supermarket $j$

##### Objective

Minimize total cost (fixed + transportation):

$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each supermarket $j \in J$,
   $$
   \sum_{i \in I} x_{ij} = d_j
   $$
2. **Supplier activation:**  
   For all $i \in I$, $j \in J$,
   $$
   x_{ij} \leq d_j y_i
   $$
   (A supplier can only supply to a supermarket if it is open.)

3. **Variable domains:**  
   $$
   y_i \in \{0,1\}, \quad x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

#### Data

- **Demand (demand.csv):**

| customer | demand |
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
| C13      | 122    |
| C14      | 66     |
| C15      | 12     |
| C16      | 21     |
| C17      | 53     |
| C18      | 105    |
| C19      | 1      |
| C20      | 253    |
| C21      | 10     |
| C22      | 53     |
| C23      | 24     |
| C24      | 122    |
| C25      | 42     |

- **Fixed costs (fixed_cost.csv):**

| supplier | fixed_costs |
|----------|-------------|
| S1       | 98.88       |
| S2       | 99.73       |
| S3       | 94.01       |
| S4       | 93.77       |
| S5       | 107.59      |
| S6       | 112.65      |
| S7       | 97.05       |
| S8       | 103         |
| S9       | 90.45       |
| S10      | 96.73       |
| S11      | 96.43       |
| S12      | 112.19      |
| S13      | 102.58      |
| S14      | 88.85       |
| S15      | 82.57       |
| S16      | 91.65       |
| S17      | 101.38      |
| S18      | 102.59      |
| S19      | 105.97      |
| S20      | 85.31       |
| S21      | 104.52      |
| S22      | 100.2       |
| S23      | 103.79      |
| S24      | 105.27      |

- **Transportation costs (transportation_costs.csv):**

Each row is a supplier $i$ (S1 to S24), each column is a supermarket $j$ (C1 to C25), and the entry is $c_{ij}$.

(For brevity, only a few rows are shown; all coefficients are available in the retrieved data.)

| supplier | C1    | C2     | C3     | ... | C25   |
|----------|-------|--------|--------|-----|-------|
| S1       | 284.11| 53.78  | 10.62  | ... | 191.46|
| S2       | 7.19  | 1031.96| 90.94  | ... | 72.39 |
| S3       | 151.1 | 884.48 | 4.33   | ... | 4.02  |
| ...      | ...   | ...    | ...    | ... | ...   |
| S24      | 79.03 | 1120.79| 31.71  | ... | 176.78|

All coefficients and identifiers are as retrieved above.

---

##### Complete Model

$$
\begin{align*}
\min\ & \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J \\
& x_{ij} \leq d_j y_i \qquad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \qquad \forall i \in I \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\end{align*}
$$

where all $f_i$, $c_{ij}$, and $d_j$ are as listed above.