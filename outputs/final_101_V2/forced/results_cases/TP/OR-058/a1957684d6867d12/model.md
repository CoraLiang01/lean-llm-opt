##### Sets

- $I = \{S1, S2, S3, S4, S5, S6\}$ (suppliers)
- $J = \{C1, C2, C3, C4, C5, C6\}$ (stores/customers)

##### Parameters

- Fixed costs:
  - $f_{S1} = 98.88$
  - $f_{S2} = 99.73$
  - $f_{S3} = 94.01$
  - $f_{S4} = 93.77$
  - $f_{S5} = 107.59$
  - $f_{S6} = 112.65$
- Transportation costs $c_{ij}$:

|      | C1     | C2     | C3      | C4      | C5     | C6     |
|------|--------|--------|---------|---------|--------|--------|
| S1   | 0.08   | 52.33  | 73.57   | 1237.33 | 0.07   | 112.16 |
| S2   | 46.02  | 175.23 | 2026.83 | 299.89  | 966.53 | 1590.42|
| S3   |1031.74 | 78.13  | 99.02   | 277.07  | 884.45 | 1800.86|
| S4   | 868.75 | 94.2   |1776.34  | 285.48  | 868.85 | 86.55  |
| S5   |1577    | 760.15 |2090.19  | 43.2    |1577.12 |1095.17 |
| S6   | 49.14  | 4.33   |2079.57  | 277.04  |1032.01 |1543.49 |

- Demands:
  - $d_{C1} = 216$
  - $d_{C2} = 216$
  - $d_{C3} = 216$
  - $d_{C4} = 144$
  - $d_{C5} = 144$
  - $d_{C6} = 144$

##### Decision Variables

- $y_i \in \{0,1\}$: $1$ if supplier $i$ is open, $0$ otherwise, for $i \in I$
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to store $j$, for $i \in I$, $j \in J$

##### Mathematical Model

Minimize total cost:
$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:

1. Demand satisfaction for each store:
$$
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
$$

2. Linking constraint: only open suppliers can supply:
$$
x_{ij} \leq M_{ij} y_i \qquad \forall i \in I,\, j \in J
$$
where $M_{ij}$ is a sufficiently large constant (e.g., $M_{ij} = d_j$).

3. Binary and non-negativity:
$$
y_i \in \{0,1\} \qquad \forall i \in I
$$
$$
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
$$

##### Data

- $I = \{S1, S2, S3, S4, S5, S6\}$
- $J = \{C1, C2, C3, C4, C5, C6\}$
- $f_{S1} = 98.88$, $f_{S2} = 99.73$, $f_{S3} = 94.01$, $f_{S4} = 93.77$, $f_{S5} = 107.59$, $f_{S6} = 112.65$
- $c_{ij}$ as in the table above
- $d_{C1} = 216$, $d_{C2} = 216$, $d_{C3} = 216$, $d_{C4} = 144$, $d_{C5} = 144$, $d_{C6} = 144$