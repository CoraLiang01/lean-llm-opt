##### Decision Variables

$x_{ij} \geq 0$: quantity of Adidas products shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. Store demand: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j\in J$
2. Supplier activation: $\sum_{j\in J} x_{ij} \leq M y_i,\quad \forall i\in I$
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

Where $M = \sum_{j\in J} d_j = 216 + 216 + 216 + 144 + 144 + 144 = 1080$.

##### Parameters

- Suppliers $I = \{S1, S2, S3, S4, S5, S6\}$
- Stores $J = \{C1, C2, C3, C4, C5, C6\}$

- Demands $d_j$:
  - $d_{C1} = 216$
  - $d_{C2} = 216$
  - $d_{C3} = 216$
  - $d_{C4} = 144$
  - $d_{C5} = 144$
  - $d_{C6} = 144$

- Fixed costs $f_i$:
  - $f_{S1} = 98.88$
  - $f_{S2} = 99.73$
  - $f_{S3} = 94.01$
  - $f_{S4} = 93.77$
  - $f_{S5} = 107.59$
  - $f_{S6} = 112.65$

- Transportation costs $c_{ij}$:

|        |   C1   |   C2   |   C3   |   C4   |   C5   |   C6   |
|--------|--------|--------|--------|--------|--------|--------|
|  S1    |  0.08  | 52.33  | 73.57  |1237.33 |  0.07  |112.16  |
|  S2    | 46.02  |175.23  |2026.83 | 299.89 | 966.53 |1590.42 |
|  S3    |1031.74 | 78.13  | 99.02  | 277.07 | 884.45 |1800.86 |
|  S4    | 868.75 | 94.20  |1776.34 | 285.48 | 868.85 | 86.55  |
|  S5    |1577.00 |760.15  |2090.19 |  43.20 |1577.12 |1095.17 |
|  S6    | 49.14  |  4.33  |2079.57 | 277.04 |1032.01 |1543.49 |

##### Complete Model

\[
\begin{align*}
\min\quad & \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i \\
\text{s.t.}\quad & \sum_{i\in I} x_{ij} = d_j,\quad \forall j\in J \\
& \sum_{j\in J} x_{ij} \leq 1080\, y_i,\quad \forall i\in I \\
& x_{ij} \geq 0,\quad \forall i\in I,\, j\in J \\
& y_i \in \{0,1\},\quad \forall i\in I
\end{align*}
\]

With all parameters as specified above.