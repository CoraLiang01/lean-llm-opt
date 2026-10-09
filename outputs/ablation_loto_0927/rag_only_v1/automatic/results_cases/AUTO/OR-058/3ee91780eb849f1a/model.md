Let:
- I = {S1, S2, S3, S4, S5, S6} be the set of suppliers.
- J = {C1, C2, C3, C4, C5, C6} be the set of stores.
- f_i = fixed cost of opening supplier i ∈ I.
- c_{ij} = transportation cost per unit from supplier i ∈ I to store j ∈ J.
- d_j = demand at store j ∈ J.

Decision variables:
- y_i ∈ {0,1}, for each i ∈ I, where y_i = 1 if supplier i is open, 0 otherwise.
- x_{ij} ≥ 0, for each i ∈ I, j ∈ J, representing the quantity supplied from supplier i to store j.

Parameters (from the CSV data):

Suppliers and Fixed Costs:
- S1: f_1 = 98.88
- S2: f_2 = 99.73
- S3: f_3 = 94.01
- S4: f_4 = 93.77
- S5: f_5 = 107.59
- S6: f_6 = 112.65

Stores and Demands:
- C1: d_1 = 216
- C2: d_2 = 216
- C3: d_3 = 216
- C4: d_4 = 144
- C5: d_5 = 144
- C6: d_6 = 144

Transportation Costs Matrix c_{ij} (rows: suppliers S1–S6, columns: stores C1–C6):

|        | C1     | C2     | C3      | C4      | C5     | C6     |
|--------|--------|--------|---------|---------|--------|--------|
| S1     | 0.08   | 52.33  | 73.57   | 1237.33 | 0.07   | 112.16 |
| S2     | 46.02  | 175.23 | 2026.83 | 299.89  | 966.53 | 1590.42|
| S3     | 1031.74| 78.13  | 99.02   | 277.07  | 884.45 | 1800.86|
| S4     | 868.75 | 94.2   | 1776.34 | 285.48  | 868.85 | 86.55  |
| S5     | 1577   | 760.15 | 2090.19 | 43.2    | 1577.12| 1095.17|
| S6     | 49.14  | 4.33   | 2079.57 | 277.04  | 1032.01| 1543.49|

Mathematical Model:

Minimize total cost:
\[
\min \sum_{i=1}^{6} f_i y_i + \sum_{i=1}^{6} \sum_{j=1}^{6} c_{ij} x_{ij}
\]
where:
- \( f_1 = 98.88, f_2 = 99.73, f_3 = 94.01, f_4 = 93.77, f_5 = 107.59, f_6 = 112.65 \)
- \( c_{ij} \) as per the matrix above.

Subject to:

1. Demand satisfaction at each store:
\[
\sum_{i=1}^{6} x_{ij} = d_j \quad \forall j \in \{1,2,3,4,5,6\}
\]
where:
- \( d_1 = 216, d_2 = 216, d_3 = 216, d_4 = 144, d_5 = 144, d_6 = 144 \)

2. Supply only from open suppliers:
\[
x_{ij} \leq d_j y_i \quad \forall i \in \{1,2,3,4,5,6\}, \forall j \in \{1,2,3,4,5,6\}
\]

3. Binary and non-negativity constraints:
\[
y_i \in \{0,1\} \quad \forall i \in \{1,2,3,4,5,6\}
\]
\[
x_{ij} \geq 0 \quad \forall i \in \{1,2,3,4,5,6\}, \forall j \in \{1,2,3,4,5,6\}
\]

Where:
- \( x_{ij} \) = quantity supplied from supplier S_i to store C_j
- \( y_i \) = 1 if supplier S_i is open, 0 otherwise

All parameters (fixed costs, demands, transportation costs) are explicitly listed above. This model ensures all store demands are met, suppliers are only used if opened (incurring their fixed cost), and the total cost (fixed + transportation) is minimized.