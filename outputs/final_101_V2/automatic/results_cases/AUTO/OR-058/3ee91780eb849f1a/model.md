##### Decision Variables

- $x_{ij} \geq 0$: Quantity of Adidas products shipped from supplier (facility) $i \in I$ to store (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Parameters

- $I = \{S1, S2, S3, S4, S5, S6\}$ (set of suppliers/facilities)
- $J = \{C1, C2, C3, C4, C5, C6\}$ (set of stores/customers)

- Demand for each store:
  - $d_{C1} = 216$
  - $d_{C2} = 216$
  - $d_{C3} = 216$
  - $d_{C4} = 144$
  - $d_{C5} = 144$
  - $d_{C6} = 144$

- Fixed cost for each supplier:
  - $f_{S1} = 98.88$
  - $f_{S2} = 99.73$
  - $f_{S3} = 94.01$
  - $f_{S4} = 93.77$
  - $f_{S5} = 107.59$
  - $f_{S6} = 112.65$

- Transportation cost per unit from each supplier to each store:

|        | C1     | C2     | C3      | C4      | C5     | C6     |
|--------|--------|--------|---------|---------|--------|--------|
| S1     | 0.08   | 52.33  | 73.57   | 1237.33 | 0.07   | 112.16 |
| S2     | 46.02  | 175.23 | 2026.83 | 299.89  | 966.53 | 1590.42|
| S3     |1031.74 | 78.13  | 99.02   | 277.07  | 884.45 | 1800.86|
| S4     | 868.75 | 94.2   |1776.34  | 285.48  | 868.85 | 86.55  |
| S5     |1577    | 760.15 |2090.19  | 43.2    |1577.12 |1095.17 |
| S6     | 49.14  | 4.33   |2079.57  | 277.04  |1032.01 |1543.49 |

Let $c_{ij}$ denote the transportation cost from supplier $i$ to store $j$ as above.

Let $M = \sum_{j \in J} d_j = 216 + 216 + 216 + 144 + 144 + 144 = 1080$ (sufficiently large upper bound for linking constraints).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier activation (linking):**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All Parameters (as retrieved)

- $I = \{S1, S2, S3, S4, S5, S6\}$
- $J = \{C1, C2, C3, C4, C5, C6\}$
- $d = \{C1:216,\, C2:216,\, C3:216,\, C4:144,\, C5:144,\, C6:144\}$
- $f = \{S1:98.88,\, S2:99.73,\, S3:94.01,\, S4:93.77,\, S5:107.59,\, S6:112.65\}$
- $c_{ij}$ as in the table above
- $M = 1080$