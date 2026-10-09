##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. Supermarket demand: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j\in J$
2. Supplier activation: $\sum_{j\in J} x_{ij} \leq M y_i,\quad \forall i\in I$
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

Where $M = \sum_{j\in J} d_j = 2287$ is a valid upper bound on total shipments from any supplier (since there are no explicit supplier capacity limits).

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

##### Parameters

- Demands $d_j$ for each supermarket $j$:
  - $d_{\text{C1}} = 1097$
  - $d_{\text{C2}} = 61$
  - $d_{\text{C3}} = 11$
  - $d_{\text{C4}} = 7$
  - $d_{\text{C5}} = 82$
  - $d_{\text{C6}} = 37$
  - $d_{\text{C7}} = 483$
  - $d_{\text{C8}} = 582$
  - $d_{\text{C9}} = 223$
  - $d_{\text{C10}} = 89$
  - $d_{\text{C11}} = 60$
  - $d_{\text{C12}} = 55$

- Fixed costs $f_i$ for each supplier $i$:
  - $f_{\text{S1}} = 98.88$
  - $f_{\text{S2}} = 99.73$
  - $f_{\text{S3}} = 94.01$
  - $f_{\text{S4}} = 93.77$
  - $f_{\text{S5}} = 107.59$
  - $f_{\text{S6}} = 112.65$
  - $f_{\text{S7}} = 97.05$
  - $f_{\text{S8}} = 103$
  - $f_{\text{S9}} = 90.45$
  - $f_{\text{S10}} = 96.73$
  - $f_{\text{S11}} = 96.43$
  - $f_{\text{S12}} = 112.19$

- Transportation costs $c_{ij}$ (supplier $i$ to supermarket $j$):

|        |  C1    |   C2   |   C3   |   C4   |   C5   |   C6   |   C7   |   C8   |   C9   |  C10   |  C11   |  C12   |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| S1     | 284.11 | 53.78  | 10.62  | 111.27 | 158.5  | 8.79   | 53.79  | 8.84   |1911.43 | 8.87   |1129.47 |185.53  |
| S2     | 7.19   |1031.96 | 90.94  | 276.97 | 0.45   | 0.20   | 49.14  | 1.05   |2079.54 | 1.45   | 49.14  | 0.05   |
| S3     |151.10  | 884.48 | 4.33   | 277.04 | 0.33   | 0.19   | 49.14  | 0.99   | 99.03  | 1.63   | 884.47 | 0.96   |
| S4     |144.16  | 868.75 | 94.20  | 285.48 | 16.93  | 0.94   | 868.78 | 16.60  | 98.69  | 19.74  | 868.74 | 19.85  |
| S5     |151.34  |1030.88 | 91.43  | 13.24  | 0.72   | 0.87   | 49.09  | 0.01   | 99.05  | 0.84   | 883.60 | 0.58   |
| S6     | 7.18   | 49.13  | 90.72  | 277.57 | 0.37   | 0.58   |1031.74 | 0.76   |1782.98 | 1.06   | 884.31 | 0.34   |
| S7     |104.38  |1324.35 |1829.39 |1857.57 |1782.69 |2079.47 |1324.31 |2080.29 | 0.00   |2080.99 |1545.08 | 99.07  |
| S8     |129.51  |1031.96 | 4.33   | 276.97 | 0.02   | 0.23   | 884.56 | 1.22   |2079.54 | 1.69   | 49.14  | 0.05   |
| S9     | 50.93  | 5.75   |1057.85 | 58.62  | 47.63  |1000.41 |103.48  | 47.60  |1642.85 | 47.59  | 5.75   |999.94  |
| S10    |129.62  | 884.35 | 91.10  | 277.12 | 0.27   | 0.07   |1031.78 | 0.91   | 99.03  | 0.08   | 49.13  | 0.04   |
| S11    | 53.30  | 0.00   | 941.91 | 58.92  |1031.61 | 49.13  | 0.03   |1030.99 |1324.29 | 49.10  | 0.08   | 49.12  |
| S12    |959.55  | 0.11   | 941.98 |1237.42 | 49.13  |1031.86 | 0.09   |1031.07 | 73.57  | 49.10  | 0.12   |1031.53 |

##### Summary

- Decision variables: $x_{ij}$ (continuous, $\geq 0$), $y_i$ (binary)
- Objective: Minimize total transportation and fixed costs
- Each supermarket's demand must be met exactly
- Suppliers can only ship if open; $M = 2287$ is a valid upper bound for all suppliers

All parameters, vectors, and matrices are as above, with identifiers and coefficients preserved from the CSV data.