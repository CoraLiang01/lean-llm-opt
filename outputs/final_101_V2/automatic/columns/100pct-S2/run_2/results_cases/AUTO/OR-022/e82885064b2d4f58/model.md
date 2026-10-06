##### Decision Variables

$x_{ij} \geq 0$: Quantity of goods shipped from supplier (facility) $i \in I$ to branch (customer) $j \in J$ (continuous).
$y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{S1, S2, S3, S4, S5\}$ (set of suppliers/facilities)
- $J = \{C1, C2, C3, C4, C5\}$ (set of branches/customers)

- Demand at each branch:
  - $d_{C1} = 143$
  - $d_{C2} = 6$
  - $d_{C3} = 10$
  - $d_{C4} = 25$
  - $d_{C5} = 3$

- Fixed opening cost for each supplier:
  - $f_{S1} = 97.65$
  - $f_{S2} = 99.76$
  - $f_{S3} = 100.76$
  - $f_{S4} = 105.32$
  - $f_{S5} = 98.88$

- Transportation cost per unit from each supplier to each branch:

|           | C1      | C2     | C3    | C4      | C5     |
|-----------|---------|--------|-------|---------|--------|
| S1        | 150.74  | 0.02   | 49.13 | 2080.15 | 426.4  |
| S2        | 233.05  | 97.73  | 49.84 | 1982.39 | 23.96  |
| S3        | 55.68   | 935.61 | 4.03  | 73.09   | 525.32 |
| S4        | 1483.82 | 1801.08|112.16 | 816.05  | 107.01 |
| S5        | 1119.47 | 884.31 | 0.08  | 1544.95 | 543.67 |

- Facility staff count (proxy for capacity, if needed):
  - $cap_{S1} = 8$
  - $cap_{S2} = 12$
  - $cap_{S3} = 50$
  - $cap_{S4} = 8$
  - $cap_{S5} = 20$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where $c_{ij}$ is the transportation cost per unit from supplier $i$ to branch $j$, and $f_i$ is the fixed opening cost for supplier $i$.

##### Constraints

1. **Demand Satisfaction:** Each branch must receive exactly its demand.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier Activation:** No goods can be shipped from a supplier unless it is open.
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i$ is a sufficiently large number (e.g., $M_i = \sum_{j \in J} d_j = 187$ for all $i$).

3. **Nonnegativity and Binary:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 187\, y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

where:

- $I = \{S1, S2, S3, S4, S5\}$
- $J = \{C1, C2, C3, C4, C5\}$
- $d_{C1} = 143$, $d_{C2} = 6$, $d_{C3} = 10$, $d_{C4} = 25$, $d_{C5} = 3$
- $f_{S1} = 97.65$, $f_{S2} = 99.76$, $f_{S3} = 100.76$, $f_{S4} = 105.32$, $f_{S5} = 98.88$
- $c_{ij}$ as in the table above
- $M_i = 187$ for all $i$ (total demand)

All parameters, vectors, and matrices are explicitly listed as retrieved.