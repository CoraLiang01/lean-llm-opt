##### Decision Variables

- $x_{ij} \geq 0$: Quantity of Adidas products shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{S1, S2, S3, S4, S5, S6\}$ (Suppliers)
- $J = \{C1, C2, C3, C4, C5, C6\}$ (Stores)
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
- Transportation costs $c_{ij}$ (per unit from supplier $i$ to store $j$):

|        | C1     | C2     | C3     | C4     | C5     | C6     |
|--------|--------|--------|--------|--------|--------|--------|
| S1     | 0.08   | 52.33  | 73.57  | 1237.33| 0.07   | 112.16 |
| S2     | 46.02  | 175.23 | 2026.83| 299.89 | 966.53 | 1590.42|
| S3     |1031.74 | 78.13  | 99.02  | 277.07 | 884.45 | 1800.86|
| S4     | 868.75 | 94.2   |1776.34 | 285.48 | 868.85 | 86.55  |
| S5     |1577    | 760.15 |2090.19 | 43.2   |1577.12 |1095.17 |
| S6     | 49.14  | 4.33   |2079.57 | 277.04 |1032.01 |1543.49 |

- $M = \sum_{j \in J} d_j = 216 + 216 + 216 + 144 + 144 + 144 = 1080$

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
2. **Supplier activation:**  
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

##### Retrieved Information

{
  "suppliers": [
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6"
  ],
  "stores": [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6"
  ],
  "demand": {
    "C1": 216,
    "C2": 216,
    "C3": 216,
    "C4": 144,
    "C5": 144,
    "C6": 144
  },
  "fixed_cost": {
    "S1": 98.88,
    "S2": 99.73,
    "S3": 94.01,
    "S4": 93.77,
    "S5": 107.59,
    "S6": 112.65
  },
  "cost": {
    "S1": {"C1": 0.08, "C2": 52.33, "C3": 73.57, "C4": 1237.33, "C5": 0.07, "C6": 112.16},
    "S2": {"C1": 46.02, "C2": 175.23, "C3": 2026.83, "C4": 299.89, "C5": 966.53, "C6": 1590.42},
    "S3": {"C1": 1031.74, "C2": 78.13, "C3": 99.02, "C4": 277.07, "C5": 884.45, "C6": 1800.86},
    "S4": {"C1": 868.75, "C2": 94.2, "C3": 1776.34, "C4": 285.48, "C5": 868.85, "C6": 86.55},
    "S5": {"C1": 1577, "C2": 760.15, "C3": 2090.19, "C4": 43.2, "C5": 1577.12, "C6": 1095.17},
    "S6": {"C1": 49.14, "C2": 4.33, "C3": 2079.57, "C4": 277.04, "C5": 1032.01, "C6": 1543.49}
  },
  "M": 1080
}