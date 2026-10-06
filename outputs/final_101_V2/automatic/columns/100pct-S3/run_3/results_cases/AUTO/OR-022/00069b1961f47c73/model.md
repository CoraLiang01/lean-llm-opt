##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier (facility) $i \in I$ to branch (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{ \text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5} \}$ (set of suppliers/facilities)
- $J = \{ \text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5} \}$ (set of branches/customers)

- Demand for each branch (from 'demand.csv'):
  - $d_{\text{C1}} = 143$
  - $d_{\text{C2}} = 6$
  - $d_{\text{C3}} = 10$
  - $d_{\text{C4}} = 25$
  - $d_{\text{C5}} = 3$

- Fixed opening cost for each supplier (from 'fixed_cost.csv'):
  - $f_{\text{S1}} = 97.65$
  - $f_{\text{S2}} = 99.76$
  - $f_{\text{S3}} = 100.76$
  - $f_{\text{S4}} = 105.32$
  - $f_{\text{S5}} = 98.88$

- Transportation cost per unit from each supplier to each branch (from 'transportation_costs.csv'):

|         | C1      | C2     | C3     | C4      | C5     |
|---------|---------|--------|--------|---------|--------|
| **S1**  | 150.74  | 0.02   | 49.13  | 2080.15 | 426.4  |
| **S2**  | 233.05  | 97.73  | 49.84  | 1982.39 | 23.96  |
| **S3**  | 55.68   | 935.61 | 4.03   | 73.09   | 525.32 |
| **S4**  | 1483.82 | 1801.08| 112.16 | 816.05  | 107.01 |
| **S5**  | 1119.47 | 884.31 | 0.08   | 1544.95 | 543.67 |

Let $c_{ij}$ denote the transportation cost per unit from supplier $i$ to branch $j$ as above.

Let $M = \sum_{j \in J} d_j = 143 + 6 + 10 + 25 + 3 = 187$ (a valid upper bound for total shipments from any supplier, since there are no explicit supplier capacity limits).

##### Objective Function

\[
\min \left( \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \right)
\]

##### Constraints

1. **Demand satisfaction at each branch:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation logic:**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   (If $y_i = 0$, then $x_{ij} = 0$ for all $j$; if $y_i = 1$, supplier $i$ can supply up to $M$ units.)

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

Where:

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$
- $d_j$ as above
- $f_i$ as above
- $c_{ij}$ as above
- $M = 187$

All parameters, vectors, and matrices are as retrieved and shown above.