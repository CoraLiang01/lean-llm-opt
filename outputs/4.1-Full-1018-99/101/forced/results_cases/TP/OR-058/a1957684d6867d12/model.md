##### Sets

- $I = \{S1, S2, S3, S4, S5, S6\}$: suppliers  
- $J = \{C1, C2, C3, C4, C5, C6\}$: stores

##### Parameters

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

- Transportation costs $c_{ij}$:

|        | C1     | C2     | C3      | C4      | C5     | C6     |
|--------|--------|--------|---------|---------|--------|--------|
| S1     | 0.08   | 52.33  | 73.57   | 1237.33 | 0.07   | 112.16 |
| S2     | 46.02  | 175.23 | 2026.83 | 299.89  | 966.53 | 1590.42|
| S3     |1031.74 | 78.13  | 99.02   | 277.07  | 884.45 |1800.86 |
| S4     | 868.75 | 94.2   |1776.34  | 285.48  | 868.85 | 86.55  |
| S5     |1577    |760.15  |2090.19  | 43.2    |1577.12 |1095.17 |
| S6     | 49.14  | 4.33   |2079.57  | 277.04  |1032.01 |1543.49 |

##### Decision Variables

- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to store $j$ (continuous)
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise (binary)

##### Objective

Minimize total cost (fixed + transportation):

$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$

2. **Activation constraint:**  
   For each supplier $i \in I$ and each store $j \in J$,
   $$
   x_{ij} \leq M_{ij} y_i
   $$
   where $M_{ij}$ is a sufficiently large constant (e.g., $M_{ij} = d_j$).

3. **Variable domains:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$
   $$
   y_i \in \{0,1\} \quad \forall i \in I
   $$

##### Complete Numerical Formulation

Minimize
$$
98.88\,y_{S1} + 99.73\,y_{S2} + 94.01\,y_{S3} + 93.77\,y_{S4} + 107.59\,y_{S5} + 112.65\,y_{S6}
+ \\
\Big[
0.08\,x_{S1,C1} + 52.33\,x_{S1,C2} + 73.57\,x_{S1,C3} + 1237.33\,x_{S1,C4} + 0.07\,x_{S1,C5} + 112.16\,x_{S1,C6} + \\
46.02\,x_{S2,C1} + 175.23\,x_{S2,C2} + 2026.83\,x_{S2,C3} + 299.89\,x_{S2,C4} + 966.53\,x_{S2,C5} + 1590.42\,x_{S2,C6} + \\
1031.74\,x_{S3,C1} + 78.13\,x_{S3,C2} + 99.02\,x_{S3,C3} + 277.07\,x_{S3,C4} + 884.45\,x_{S3,C5} + 1800.86\,x_{S3,C6} + \\
868.75\,x_{S4,C1} + 94.2\,x_{S4,C2} + 1776.34\,x_{S4,C3} + 285.48\,x_{S4,C4} + 868.85\,x_{S4,C5} + 86.55\,x_{S4,C6} + \\
1577\,x_{S5,C1} + 760.15\,x_{S5,C2} + 2090.19\,x_{S5,C3} + 43.2\,x_{S5,C4} + 1577.12\,x_{S5,C5} + 1095.17\,x_{S5,C6} + \\
49.14\,x_{S6,C1} + 4.33\,x_{S6,C2} + 2079.57\,x_{S6,C3} + 277.04\,x_{S6,C4} + 1032.01\,x_{S6,C5} + 1543.49\,x_{S6,C6}
\Big]
$$

Subject to:

For each store:
- $\sum_{i \in I} x_{i,C1} \geq 216$
- $\sum_{i \in I} x_{i,C2} \geq 216$
- $\sum_{i \in I} x_{i,C3} \geq 216$
- $\sum_{i \in I} x_{i,C4} \geq 144$
- $\sum_{i \in I} x_{i,C5} \geq 144$
- $\sum_{i \in I} x_{i,C6} \geq 144$

For each supplier and store:
- $x_{ij} \leq d_j\, y_i$ for all $i \in I$, $j \in J$

Variable domains:
- $x_{ij} \geq 0$ for all $i \in I$, $j \in J$
- $y_i \in \{0,1\}$ for all $i \in I$