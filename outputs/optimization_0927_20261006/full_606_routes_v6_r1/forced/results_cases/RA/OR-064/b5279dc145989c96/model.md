Let:
- $S = \{$S1, S2, ..., S24$\}$ be the set of suppliers (from fixed_cost.csv and transportation_costs.csv, using "Unnamed: 0" as supplier ID).
- $C = \{$C1, C2, ..., C25$\}$ be the set of supermarkets/customers (from demand.csv and transportation_costs.csv).
- $f_i$ be the fixed cost of opening supplier $i \in S$ (from fixed_cost.csv).
- $t_{ij}$ be the transportation cost per unit from supplier $i \in S$ to customer $j \in C$ (from transportation_costs.csv).
- $d_j$ be the demand of customer $j \in C$ (from demand.csv).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise.
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to customer $j$.

#### Objective:
Minimize total cost (fixed + transportation):
$$
\min \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} t_{ij} x_{ij}
$$

#### Constraints:

1. **Demand satisfaction for each customer:**
   $$
   \sum_{i \in S} x_{ij} = d_j \quad \forall j \in C
   $$

2. **Suppliers can only supply if open:**
   $$
   x_{ij} \leq d_j y_i \quad \forall i \in S, \forall j \in C
   $$

3. **Variable domains:**
   $$
   y_i \in \{0,1\} \quad \forall i \in S
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in S, \forall j \in C
   $$

---

#### Data

**Demand (demand.csv, in source order):**
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

**Fixed costs (fixed_cost.csv, in source order):**
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

**Transportation costs (transportation_costs.csv, in source order):**

Each row is a supplier, each column is a customer. For example, the cost from S1 to C1 is 284.11, from S1 to C2 is 53.78, etc.

| Supplier | C1    | C2    | C3    | ... | C25   |
|----------|-------|-------|-------|-----|-------|
| S1       | 284.11| 53.78 | 10.62 | ... | 191.46|
| S2       | 7.19  |1031.96| 90.94 | ... | 72.39 |
| ...      | ...   | ...   | ...   | ... | ...   |
| S24      | 79.03 |1120.79| 31.71 | ... |176.78 |

(Full matrix as in the retrieved data above.)

---

**Indices:**
- $i \in S = \{$S1, S2, ..., S24$\}$
- $j \in C = \{$C1, C2, ..., C25$\}$

**Parameters:**
- $f_i$: fixed cost for supplier $i$ (see table above)
- $t_{ij}$: transportation cost per unit from $i$ to $j$ (see matrix above)
- $d_j$: demand for customer $j$ (see table above)

**Variables:**
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from $i$ to $j$

---

**Complete Model:**

$$
\begin{align*}
\min \quad & \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} t_{ij} x_{ij} \\
\text{s.t.} \quad & \sum_{i \in S} x_{ij} = d_j \quad \forall j \in C \\
& x_{ij} \leq d_j y_i \quad \forall i \in S, \forall j \in C \\
& y_i \in \{0,1\} \quad \forall i \in S \\
& x_{ij} \geq 0 \quad \forall i \in S, \forall j \in C \\
\end{align*}
$$

All coefficients and identifiers are as retrieved and preserved in source order.