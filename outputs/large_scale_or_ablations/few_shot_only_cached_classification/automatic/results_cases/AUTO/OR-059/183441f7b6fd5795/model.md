##### Decision Variables

$x_{ij} \geq 0$: Number of vehicles shipped from supplier $i \in I$ to dealership $j \in J$ (continuous).
$y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. Dealership demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Supplier activation:
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 41,\!091,\!073,\!000$ (sum of all dealership demands).
3. Domains:
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Parameters

- Suppliers $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}\}$
- Dealerships $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}\}$

- Demands $d_j$:
  - $d_{\text{C1}} = 4,\!742,\!532,\!000$
  - $d_{\text{C2}} = 1,\!600,\!594,\!000$
  - $d_{\text{C3}} = 5,\!086,\!889,\!000$
  - $d_{\text{C4}} = 1,\!027,\!326,\!000$
  - $d_{\text{C5}} = 11,\!926,\!044,\!000$
  - $d_{\text{C6}} = 9,\!058,\!407,\!000$
  - $d_{\text{C7}} = 5,\!344,\!367,\!000$
  - $d_{\text{C8}} = 677,\!201,\!000$
  - $d_{\text{C9}} = 3,\!236,\!493,\!000$

- Fixed costs $f_i$:
  - $f_{\text{S1}} = 100.64$
  - $f_{\text{S2}} = 98.72$
  - $f_{\text{S3}} = 100.18$
  - $f_{\text{S4}} = 96.58$
  - $f_{\text{S5}} = 95.75$
  - $f_{\text{S6}} = 99.06$
  - $f_{\text{S7}} = 101.78$
  - $f_{\text{S8}} = 93.86$

- Transportation costs $c_{ij}$ (per vehicle):

|        |   C1    |   C2    |   C3    |   C4    |   C5    |   C6    |   C7    |   C8    |   C9    |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| S1     | 1091.04 |   85.72 |   99.08 |  747.35 |  893.86 |   23.65 |   15.11 |   15.03 |  497.88 |
| S2     |   58.88 | 1617.16 | 1786.44 |  951.81 |   56.45 |  642.77 |   16.69 |    0.63 |   11.20 |
| S3     |  110.47 |    0.04 |   38.89 | 1397.95 | 2361.45 |  107.62 | 1598.50 |   76.41 | 1382.84 |
| S4     | 1458.85 | 1049.27 |  597.32 | 1731.90 |   69.09 | 1227.17 | 1187.55 | 1017.16 |   52.15 |
| S5     |    0.38 | 2315.52 | 1313.06 | 1253.71 |   50.24 |   29.19 |   60.17 | 1077.35 |   70.11 |
| S6     |   58.20 | 1395.81 |   84.60 |  830.64 | 1003.86 |  631.17 |   31.13 |    1.40 |  246.24 |
| S7     | 1255.23 | 1382.31 |   78.79 |  829.02 |   67.31 |  877.35 |  185.28 |  221.98 |    0.05 |
| S8     | 1990.09 |    1.23 |   38.97 | 1396.35 |  112.54 |  107.54 | 1596.74 |   76.32 | 1183.79 |

- $M = 41,\!091,\!073,\!000$ (sum of all dealership demands)

##### Model Summary

Minimize total transportation and supplier opening costs, ensuring all dealership demands are met, and only open suppliers can ship vehicles. All parameters and data are as above.