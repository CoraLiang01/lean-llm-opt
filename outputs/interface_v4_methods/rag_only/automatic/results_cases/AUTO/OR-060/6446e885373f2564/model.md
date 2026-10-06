Mathematical Model for the Facility Location and Allocation Problem

Sets:
- Let I = {S1, S2, ..., S12} be the set of suppliers (facilities).
- Let J = {C1, C2, ..., C12} be the set of supermarkets (customers).

Parameters:
- Fixed cost for opening supplier i ∈ I: 
  - f = [f_S1, f_S2, ..., f_S12] = [98.88, 99.73, 94.01, 93.77, 107.59, 112.65, 97.05, 103, 90.45, 96.73, 96.43, 112.19]
- Transportation cost per unit from supplier i to customer j: 
  - c_ij is the 12×12 matrix below (rows: S1–S12, columns: C1–C12):

|        | C1     | C2     | C3     | C4     | C5     | C6     | C7     | C8     | C9     | C10    | C11    | C12    |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| S1     | 284.11 | 53.78  | 10.62  | 111.27 | 158.5  | 8.79   | 53.79  | 8.84   | 1911.43| 8.87   | 1129.47| 185.53 |
| S2     | 7.19   | 1031.96| 90.94  | 276.97 | 0.45   | 0.2    | 49.14  | 1.05   | 2079.54| 1.45   | 49.14  | 0.05   |
| S3     | 151.1  | 884.48 | 4.33   | 277.04 | 0.33   | 0.19   | 49.14  | 0.99   | 99.03  | 1.63   | 884.47 | 0.96   |
| S4     | 144.16 | 868.75 | 94.2   | 285.48 | 16.93  | 0.94   | 868.78 | 16.6   | 98.69  | 19.74  | 868.74 | 19.85  |
| S5     | 151.34 | 1030.88| 91.43  | 13.24  | 0.72   | 0.87   | 49.09  | 0.01   | 99.05  | 0.84   | 883.6  | 0.58   |
| S6     | 7.18   | 49.13  | 90.72  | 277.57 | 0.37   | 0.58   | 1031.74| 0.76   | 1782.98| 1.06   | 884.31 | 0.34   |
| S7     | 104.38 | 1324.35| 1829.39| 1857.57| 1782.69| 2079.47| 1324.31| 2080.29| 0      | 2080.99| 1545.08| 99.07  |
| S8     | 129.51 | 1031.96| 4.33   | 276.97 | 0.02   | 0.23   | 884.56 | 1.22   | 2079.54| 1.69   | 49.14  | 0.05   |
| S9     | 50.93  | 5.75   | 1057.85| 58.62  | 47.63  | 1000.41| 103.48 | 47.6   | 1642.85| 47.59  | 5.75   | 999.94 |
| S10    | 129.62 | 884.35 | 91.10  | 277.12 | 0.27   | 0.07   | 1031.78| 0.91   | 99.03  | 0.08   | 49.13  | 0.04   |
| S11    | 53.3   | 0      | 941.91 | 58.92  | 1031.61| 49.13  | 0.03   | 1030.99| 1324.29| 49.1   | 0.08   | 49.12  |
| S12    | 959.55 | 0.11   | 941.98 | 1237.42| 49.13  | 1031.86| 0.09   | 1031.07| 73.57  | 49.1   | 0.12   | 1031.53|

- Demand for each customer j ∈ J:
  - d = [d_C1, d_C2, ..., d_C12] = [1097, 61, 11, 7, 82, 37, 483, 582, 223, 89, 60, 55]

Decision Variables:
- y_i ∈ {0,1} for i ∈ I: 1 if supplier i is open, 0 otherwise.
- x_ij ≥ 0 for i ∈ I, j ∈ J: quantity supplied from supplier i to customer j.

Mathematical Formulation:

Objective:
Minimize the total cost (fixed + transportation):

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

where:
- \( f_i \) is the fixed cost for supplier i (see vector above),
- \( c_{ij} \) is the transportation cost per unit from supplier i to customer j (see matrix above),
- \( x_{ij} \) is the quantity supplied from i to j,
- \( y_i \) is the binary variable indicating if supplier i is open.

Subject to:

1. Demand satisfaction for each customer:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]

2. Supply only from open suppliers:
\[
x_{ij} \leq d_j y_i \quad \forall i \in I, \forall j \in J
\]

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I, \forall j \in J
\]

Where all parameters (fixed costs, transportation costs, and demands) are as listed above. This model determines which suppliers to open and how to allocate supply to minimize the total cost while meeting all supermarket demands.