Let:
- $S$ = set of suppliers: $\{S1, S2, S3, S4, S5, S6\}$
- $C$ = set of customers (stores): $\{C1, C2, C3, C4, C5, C6\}$
- $f_i$ = fixed cost of opening supplier $i$
- $t_{ij}$ = transportation cost per unit from supplier $i$ to customer $j$
- $d_j$ = demand at customer $j$
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to customer $j$ (integer, but can be continuous if not otherwise specified)

#### Data

- Fixed costs ($f_i$):

| Supplier | Fixed Cost |
|----------|------------|
| S1       | 98.88      |
| S2       | 99.73      |
| S3       | 94.01      |
| S4       | 93.77      |
| S5       | 107.59     |
| S6       | 112.65     |

- Demands ($d_j$):

| Customer | Demand |
|----------|--------|
| C1       | 216    |
| C2       | 216    |
| C3       | 216    |
| C4       | 144    |
| C5       | 144    |
| C6       | 144    |

- Transportation costs ($t_{ij}$):

|        | C1     | C2     | C3     | C4     | C5     | C6     |
|--------|--------|--------|--------|--------|--------|--------|
| S1     | 0.08   | 52.33  | 73.57  | 1237.33| 0.07   | 112.16 |
| S2     | 46.02  | 175.23 | 2026.83| 299.89 | 966.53 | 1590.42|
| S3     |1031.74 | 78.13  | 99.02  | 277.07 | 884.45 | 1800.86|
| S4     | 868.75 | 94.2   |1776.34 | 285.48 | 868.85 | 86.55  |
| S5     |1577    | 760.15 |2090.19 | 43.2   |1577.12 |1095.17 |
| S6     | 49.14  | 4.33   |2079.57 | 277.04 |1032.01 |1543.49 |

#### Decision Variables

- $y_i \in \{0,1\}$ for all $i \in S$
- $x_{ij} \geq 0$ for all $i \in S$, $j \in C$

#### Objective

Minimize total cost (fixed + transportation):

$$
\min \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} t_{ij} x_{ij}
$$

#### Constraints

1. **Demand satisfaction:** For each customer $j$,
   $$
   \sum_{i \in S} x_{ij} = d_j \qquad \forall j \in C
   $$

2. **Supplier activation:** For each supplier $i$ and customer $j$,
   $$
   x_{ij} \leq d_j y_i \qquad \forall i \in S, \forall j \in C
   $$
   (A supplier can only supply if it is open; $d_j$ is a valid upper bound.)

3. **Variable domains:**
   $$
   y_i \in \{0,1\} \qquad \forall i \in S
   $$
   $$
   x_{ij} \geq 0 \qquad \forall i \in S, \forall j \in C
   $$

#### Complete Model (with all coefficients):

Minimize
$$
98.88\,y_{S1} + 99.73\,y_{S2} + 94.01\,y_{S3} + 93.77\,y_{S4} + 107.59\,y_{S5} + 112.65\,y_{S6}
+ \sum_{i \in S} \sum_{j \in C} t_{ij} x_{ij}
$$
where $t_{ij}$ is as in the table above.

Subject to, for each customer:
- $\sum_{i \in S} x_{i,C1} = 216$
- $\sum_{i \in S} x_{i,C2} = 216$
- $\sum_{i \in S} x_{i,C3} = 216$
- $\sum_{i \in S} x_{i,C4} = 144$
- $\sum_{i \in S} x_{i,C5} = 144$
- $\sum_{i \in S} x_{i,C6} = 144$

For each supplier and customer:
- $x_{ij} \leq d_j y_i$ for all $i \in S$, $j \in C$

Variable domains:
- $y_i \in \{0,1\}$ for all $i \in S$
- $x_{ij} \geq 0$ for all $i \in S$, $j \in C$