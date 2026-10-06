Let:
- I = {F1, F2, ..., F15} be the set of candidate plants (facilities), indexed by i.
- J = {C1, C2, ..., C15} be the set of customers, indexed by j.

Parameters (from cost.csv and demand.csv):

Fixed opening costs for each plant (currency units):
- f = [11250, 13480, 14870, 10290, 16740, 13960, 12680, 17890, 10950, 15320, 11830, 14110, 15970, 13140, 10580]
  - f_i is the fixed cost for plant i, in order F1–F15.

Capacities for each plant (units):
- cap = [101, 124, 139, 86, 157, 133, 118, 162, 92, 144, 107, 129, 151, 113, 85]
  - cap_i is the capacity for plant i, in order F1–F15.

Per-unit transport cost matrix c_{ij} (currency units per unit shipped from plant i to customer j):

Let C be the 15x15 matrix where C[i][j] is the per-unit transport cost from plant F(i+1) to customer C(j+1):

C =

|      | C1  | C2  | C3  | C4  | C5  | C6  | C7  | C8  | C9  | C10 | C11 | C12 | C13 | C14 | C15 |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| F1   | 7.8 | 7.6 | 6.7 | 7.9 | 8.1 | 8.3 | 7.3 | 8.2 | 8.1 | 8.2 | 7.3 | 7.7 | 6.7 | 7.1 | 7.9 |
| F2   | 5.3 | 6.0 | 5.0 | 6.4 | 5.9 | 6.2 | 5.6 | 6.1 | 6.3 | 6.1 | 5.0 | 5.6 | 5.3 | 4.9 | 6.3 |
| F3   | 7.2 | 8.1 | 7.4 | 8.8 | 8.5 | 8.7 | 7.7 | 8.7 | 8.9 | 8.5 | 7.2 | 7.7 | 7.1 | 7.6 | 8.4 |
| F4   | 7.0 | 7.1 | 6.5 | 7.9 | 7.4 | 7.7 | 6.7 | 7.9 | 7.8 | 7.3 | 6.8 | 7.0 | 6.5 | 6.7 | 7.6 |
| F5   | 3.5 | 3.8 | 2.9 | 4.3 | 3.6 | 3.9 | 3.2 | 4.3 | 4.5 | 4.0 | 3.2 | 4.0 | 2.9 | 3.4 | 3.9 |
| F6   | 8.2 | 8.6 | 7.9 | 9.5 | 8.5 | 9.3 | 8.5 | 9.4 | 9.0 | 9.2 | 8.1 | 8.7 | 7.9 | 8.5 | 9.0 |
| F7   | 6.9 | 7.6 | 6.8 | 8.4 | 8.0 | 8.0 | 7.6 | 8.0 | 8.1 | 7.8 | 6.9 | 7.1 | 7.0 | 6.9 | 7.5 |
| F8   | 6.9 | 7.8 | 7.1 | 8.7 | 8.6 | 8.2 | 7.2 | 7.9 | 8.4 | 7.9 | 7.0 | 7.4 | 6.8 | 7.3 | 8.0 |
| F9   | 3.5 | 3.8 | 2.8 | 4.4 | 4.2 | 4.8 | 3.8 | 5.0 | 4.5 | 4.1 | 3.2 | 3.7 | 3.7 | 3.2 | 4.5 |
| F10  | 5.2 | 6.1 | 5.1 | 6.3 | 6.1 | 6.0 | 5.6 | 6.5 | 6.2 | 5.9 | 5.3 | 6.1 | 5.1 | 5.2 | 6.2 |
| F11  | 5.2 | 5.5 | 4.5 | 6.2 | 5.7 | 6.1 | 5.1 | 5.8 | 5.7 | 6.2 | 5.2 | 5.2 | 4.5 | 5.1 | 5.4 |
| F12  | 7.8 | 8.7 | 7.6 | 9.0 | 8.6 | 9.0 | 8.5 | 9.3 | 9.3 | 8.4 | 7.9 | 8.2 | 7.4 | 7.6 | 8.7 |
| F13  | 6.7 | 6.6 | 6.1 | 7.3 | 7.1 | 7.5 | 6.7 | 8.0 | 7.6 | 7.2 | 6.3 | 6.9 | 6.2 | 6.0 | 7.2 |
| F14  | 7.5 | 8.6 | 7.6 | 8.2 | 8.0 | 7.9 | 7.5 | 8.7 | 8.8 | 8.1 | 7.2 | 7.3 | 7.0 | 7.0 | 8.0 |
| F15  | 5.1 | 5.8 | 4.6 | 5.9 | 6.5 | 5.9 | 5.2 | 7.0 | 7.1 | 5.9 | 5.1 | 5.8 | 5.4 | 4.9 | 6.0 |

Customer demands (units):

- d = [83, 76, 91, 68, 104, 97, 88, 73, 109, 95, 82, 67, 113, 79, 92]
  - d_j is the demand for customer j, in order C1–C15.

Decision variables:
- y_i ∈ {0,1} for i ∈ I: 1 if plant i is built (opened), 0 otherwise.
- x_{ij} ≥ 0 for i ∈ I, j ∈ J: amount shipped from plant i to customer j.

Mathematical Model:

Minimize total cost:
\[
\text{Minimize} \quad Z = \sum_{i=1}^{15} f_i y_i + \sum_{i=1}^{15} \sum_{j=1}^{15} c_{ij} x_{ij}
\]
where:
- \( f_i \) is the fixed opening cost for plant i (see vector above)
- \( c_{ij} \) is the per-unit transport cost from plant i to customer j (see matrix above)
- \( y_i \) is the binary variable for opening plant i
- \( x_{ij} \) is the amount shipped from plant i to customer j

Subject to:

1. Demand satisfaction for each customer:
\[
\sum_{i=1}^{15} x_{ij} = d_j \quad \forall j = 1, ..., 15
\]
where \( d_j \) is the demand for customer j (see vector above).

2. Plant capacity (only if plant is opened):
\[
\sum_{j=1}^{15} x_{ij} \leq cap_i \cdot y_i \quad \forall i = 1, ..., 15
\]
where \( cap_i \) is the capacity of plant i (see vector above).

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i = 1, ..., 15
\]
\[
x_{ij} \geq 0 \quad \forall i = 1, ..., 15; \; j = 1, ..., 15
\]

All parameters (fixed costs, capacities, transport costs, and demands) are explicitly listed above.

This is a standard capacitated facility location problem (CFLP) with fixed opening costs and per-unit transportation costs, formulated with all explicit data from the provided CSV files.