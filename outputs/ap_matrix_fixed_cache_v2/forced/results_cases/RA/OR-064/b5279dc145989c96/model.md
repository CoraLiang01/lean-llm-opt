**Sets and Indices:**

- Let $I$ be the set of suppliers, indexed by $i$ (from fixed_cost.csv, $I = \{\text{S1}, \text{S2}, \ldots, \text{S24}\}$).
- Let $J$ be the set of supermarkets/customers, indexed by $j$ (from demand.csv, $J = \{\text{C1}, \text{C2}, \ldots, \text{C25}\}$).

**Parameters:**

- $f_i$: Fixed cost of opening supplier $i$ (from fixed_cost.csv).
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv).
- $d_j$: Demand of customer $j$ (from demand.csv).

**Decision Variables:**

- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise.
- $x_{ij} \geq 0$: Quantity supplied from supplier $i$ to customer $j$.

---

**Objective:**

Minimize total cost (fixed + transportation):

$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

---

**Constraints:**

1. **Demand Satisfaction:**  
   For each customer $j \in J$,
   $$
   \sum_{i \in I} x_{ij} = d_j
   $$

2. **Supplier Activation:**  
   For each supplier $i \in I$ and each customer $j \in J$,
   $$
   x_{ij} \leq d_j y_i
   $$
   (A supplier can only supply to a customer if it is open.)

3. **Variable Domains:**  
   $$
   y_i \in \{0,1\} \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   $$

---

**Data (in source order):**

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

- **Fixed Costs (fixed_cost.csv):**

| Supplier | fixed_costs |
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

- **Transportation Costs (transportation_costs.csv):**

Each row corresponds to a supplier $i$ (S1 to S24), each column (C1 to C25) is a customer $j$. The value is $c_{ij}$.

(For brevity, only a few rows are shown here; all data from the observation is included in the model.)

| Supplier | C1    | C2    | ... | C25   |
|----------|-------|-------|-----|-------|
| S1       | 284.11| 53.78 | ... | 191.46|
| S2       | 7.19  |1031.96| ... | 72.39 |
| ...      | ...   | ...   | ... | ...   |
| S24      | 79.03 |1120.79| ... |176.78 |

---

**Complete Model:**

$$
\begin{align*}
\min \quad & \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& x_{ij} \leq d_j y_i \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I, j \in J \\
\end{align*}
$$

where all parameters and indices are as defined above, and all data is as retrieved and listed.