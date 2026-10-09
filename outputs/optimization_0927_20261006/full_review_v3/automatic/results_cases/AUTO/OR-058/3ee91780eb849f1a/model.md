##### Decision Variables

- $x_{ij} \geq 0$: Quantity of Adidas products shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- Suppliers $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}\}$
- Stores $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}\}$

- Demand $d_j$ for each store $j$:
  - $d_{\text{C1}} = 216$
  - $d_{\text{C2}} = 216$
  - $d_{\text{C3}} = 216$
  - $d_{\text{C4}} = 144$
  - $d_{\text{C5}} = 144$
  - $d_{\text{C6}} = 144$

- Fixed cost $f_i$ for each supplier $i$:
  - $f_{\text{S1}} = 98.88$
  - $f_{\text{S2}} = 99.73$
  - $f_{\text{S3}} = 94.01$
  - $f_{\text{S4}} = 93.77$
  - $f_{\text{S5}} = 107.59$
  - $f_{\text{S6}} = 112.65$

- Transportation cost $c_{ij}$ from supplier $i$ to store $j$:

|        |  C1    |   C2   |   C3   |   C4   |   C5   |   C6   |
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

1. **Demand satisfaction:** Each store's demand must be met:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:** Suppliers can only ship if activated:
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Full Parameter Listing

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}\}$
- $d = \{216, 216, 216, 144, 144, 144\}$
- $f = \{98.88, 99.73, 94.01, 93.77, 107.59, 112.65\}$
- $C =$
  - $c_{\text{S1},j} = \{0.08, 52.33, 73.57, 1237.33, 0.07, 112.16\}$
  - $c_{\text{S2},j} = \{46.02, 175.23, 2026.83, 299.89, 966.53, 1590.42\}$
  - $c_{\text{S3},j} = \{1031.74, 78.13, 99.02, 277.07, 884.45, 1800.86\}$
  - $c_{\text{S4},j} = \{868.75, 94.2, 1776.34, 285.48, 868.85, 86.55\}$
  - $c_{\text{S5},j} = \{1577, 760.15, 2090.19, 43.2, 1577.12, 1095.17\}$
  - $c_{\text{S6},j} = \{49.14, 4.33, 2079.57, 277.04, 1032.01, 1543.49\}$
- $M = 1080$