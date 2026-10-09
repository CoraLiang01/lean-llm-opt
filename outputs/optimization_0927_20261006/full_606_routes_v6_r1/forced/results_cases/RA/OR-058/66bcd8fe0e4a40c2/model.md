Let:
- $S = \{S1, S2, S3, S4, S5, S6\}$ be the set of suppliers.
- $C = \{C1, C2, C3, C4, C5, C6\}$ be the set of customers (stores).
- $f_i$ be the fixed cost of opening supplier $i$.
- $t_{ij}$ be the transportation cost per unit from supplier $i$ to customer $j$.
- $d_j$ be the demand of customer $j$.
- $y_i \in \{0,1\}$ indicates if supplier $i$ is open.
- $x_{ij} \geq 0$ is the quantity shipped from supplier $i$ to customer $j$.

#### Data

- Demands:
  - $d_{C1} = 216$
  - $d_{C2} = 216$
  - $d_{C3} = 216$
  - $d_{C4} = 144$
  - $d_{C5} = 144$
  - $d_{C6} = 144$

- Fixed costs:
  - $f_{S1} = 98.88$
  - $f_{S2} = 99.73$
  - $f_{S3} = 94.01$
  - $f_{S4} = 93.77$
  - $f_{S5} = 107.59$
  - $f_{S6} = 112.65$

- Transportation costs $t_{ij}$:

|        | C1     | C2     | C3      | C4      | C5     | C6     |
|--------|--------|--------|---------|---------|--------|--------|
| S1     | 0.08   | 52.33  | 73.57   | 1237.33 | 0.07   | 112.16 |
| S2     | 46.02  | 175.23 | 2026.83 | 299.89  | 966.53 | 1590.42|
| S3     |1031.74 | 78.13  | 99.02   | 277.07  | 884.45 |1800.86 |
| S4     | 868.75 | 94.2   |1776.34  | 285.48  | 868.85 | 86.55  |
| S5     |1577    | 760.15 |2090.19  | 43.2    |1577.12 |1095.17 |
| S6     | 49.14  | 4.33   |2079.57  | 277.04  |1032.01 |1543.49 |

#### Decision Variables

- $y_i \in \{0,1\}$ for all $i \in S$
- $x_{ij} \geq 0$ for all $i \in S$, $j \in C$

#### Objective

Minimize total cost (fixed + transportation):

$$
\min \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} t_{ij} x_{ij}
$$

#### Constraints

1. **Demand satisfaction:** For each customer $j \in C$,
   $$
   \sum_{i \in S} x_{ij} = d_j
   $$

2. **Supplier activation:** For each supplier $i \in S$ and customer $j \in C$,
   $$
   x_{ij} \leq d_j y_i
   $$
   (A supplier can only ship to a customer if it is open; $d_j$ is a valid upper bound since no customer can receive more than its demand.)

3. **Variable domains:**
   $$
   y_i \in \{0,1\} \quad \forall i \in S
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in S, j \in C
   $$

#### Complete Model

$$
\begin{align*}
\min \quad & \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} t_{ij} x_{ij} \\
\text{s.t.} \quad & \sum_{i \in S} x_{ij} = d_j \quad \forall j \in C \\
& x_{ij} \leq d_j y_i \quad \forall i \in S, \forall j \in C \\
& y_i \in \{0,1\} \quad \forall i \in S \\
& x_{ij} \geq 0 \quad \forall i \in S, \forall j \in C \\
\end{align*}
$$

Where:

- $S = \{S1, S2, S3, S4, S5, S6\}$
- $C = \{C1, C2, C3, C4, C5, C6\}$
- $f_i$, $t_{ij}$, $d_j$ as given above.