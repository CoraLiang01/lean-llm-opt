##### Decision Variables

$x_{ij} \geq 0$: Quantity of goods shipped from supplier (facility) $i \in I$ to branch (customer) $j \in J$ (continuous).  
$y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{S1, S2, S3, S4, S5\}$ (facilities, in source order)
- $J = \{C1, C2, C3, C4, C5\}$ (branches/customers, in source order)
- Demands $d_j$ (units required at each branch):
  - $d_{C1} = 143$
  - $d_{C2} = 6$
  - $d_{C3} = 10$
  - $d_{C4} = 25$
  - $d_{C5} = 3$
- Fixed opening costs $f_i$ for each facility:
  - $f_{S1} = 97.65$
  - $f_{S2} = 99.76$
  - $f_{S3} = 100.76$
  - $f_{S4} = 105.32$
  - $f_{S5} = 98.88$
- Transportation costs $c_{ij}$ (per unit from facility $i$ to customer $j$):

|        | C1      | C2      | C3     | C4      | C5     |
|--------|---------|---------|--------|---------|--------|
| S1     | 150.74  | 0.02    | 49.13  | 2080.15 | 426.4  |
| S2     | 233.05  | 97.73   | 49.84  | 1982.39 | 23.96  |
| S3     | 55.68   | 935.61  | 4.03   | 73.09   | 525.32 |
| S4     | 1483.82 | 1801.08 | 112.16 | 816.05  | 107.01 |
| S5     | 1119.47 | 884.31  | 0.08   | 1544.95 | 543.67 |

- $M = \sum_{j \in J} d_j = 143 + 6 + 10 + 25 + 3 = 187$ (big-M for linking $x_{ij}$ and $y_i$; since there are no explicit facility capacities, this is a valid upper bound for each facility's total shipment).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each branch:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Facility activation (no shipment from closed facilities):**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### All Parameters (as retrieved)

- Facilities: $I = \{S1, S2, S3, S4, S5\}$
- Customers: $J = \{C1, C2, C3, C4, C5\}$
- Demands: $d_{C1} = 143$, $d_{C2} = 6$, $d_{C3} = 10$, $d_{C4} = 25$, $d_{C5} = 3$
- Fixed costs: $f_{S1} = 97.65$, $f_{S2} = 99.76$, $f_{S3} = 100.76$, $f_{S4} = 105.32$, $f_{S5} = 98.88$
- Transportation costs matrix $[c_{ij}]$ (rows: $I$, columns: $J$):

\[
\begin{array}{c|ccccc}
      & C1      & C2      & C3     & C4      & C5     \\
\hline
S1 & 150.74  & 0.02    & 49.13  & 2080.15 & 426.4  \\
S2 & 233.05  & 97.73   & 49.84  & 1982.39 & 23.96  \\
S3 & 55.68   & 935.61  & 4.03   & 73.09   & 525.32 \\
S4 & 1483.82 & 1801.08 & 112.16 & 816.05  & 107.01 \\
S5 & 1119.47 & 884.31  & 0.08   & 1544.95 & 543.67 \\
\end{array}
\]

- $M = 187$

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
                  & \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
                  & x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
                  & y_i \in \{0,1\}, \quad \forall i \in I \\
\end{align*}
\]

with all parameters and data as listed above.