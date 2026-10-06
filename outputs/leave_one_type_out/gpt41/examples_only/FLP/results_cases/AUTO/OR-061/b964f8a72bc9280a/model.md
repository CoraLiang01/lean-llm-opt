Let:
- There are 5 suppliers: S1, S2, S3, S4, S5.
- There are 5 branches/customers: C1, C2, C3, C4, C5.

Parameters:
- Fixed costs for opening each supplier:
    - f = [97.65, 99.76, 100.76, 105.32, 98.88], where f_i is the fixed cost for supplier S_i (i = 1,...,5).
- Demand at each branch:
    - d = [143, 6, 10, 25, 3], where d_j is the demand at branch C_j (j = 1,...,5).
- Transportation cost per unit from supplier S_i to branch C_j:
    - c = 
      |         | C1     | C2     | C3     | C4     | C5     |
      |---------|--------|--------|--------|--------|--------|
      | S1      | 150.74 | 0.02   | 49.13  | 2080.15| 426.40 |
      | S2      | 233.05 | 97.73  | 49.84  | 1982.39| 23.96  |
      | S3      | 55.68  | 935.61 | 4.03   | 73.09  | 525.32 |
      | S4      | 1483.82| 1801.08| 112.16 | 816.05 | 107.01 |
      | S5      | 1119.47| 884.31 | 0.08   | 1544.95| 543.67 |

Decision Variables:
- y_i ∈ {0,1}, for i = 1,...,5: y_i = 1 if supplier S_i is open, 0 otherwise.
- x_{ij} ≥ 0, for i = 1,...,5, j = 1,...,5: quantity of goods supplied from S_i to C_j.

Mathematical Model:

Objective:
Minimize total cost (fixed + transportation):
\[
\min \sum_{i=1}^5 f_i y_i + \sum_{i=1}^5 \sum_{j=1}^5 c_{ij} x_{ij}
\]
where:
- \( f = [97.65, 99.76, 100.76, 105.32, 98.88] \)
- \( c_{ij} \) is the transportation cost matrix as above.

Subject to:
1. Demand satisfaction at each branch:
\[
\sum_{i=1}^5 x_{ij} = d_j, \quad \forall j = 1,...,5
\]
where \( d = [143, 6, 10, 25, 3] \).

2. Supply only from open suppliers:
\[
x_{ij} \leq d_j y_i, \quad \forall i = 1,...,5, \; j = 1,...,5
\]

3. Binary and non-negativity constraints:
\[
y_i \in \{0,1\}, \quad \forall i = 1,...,5
\]
\[
x_{ij} \geq 0, \quad \forall i = 1,...,5, \; j = 1,...,5
\]

Full parameter listing:
- Suppliers: S = {S1, S2, S3, S4, S5}
- Branches: C = {C1, C2, C3, C4, C5}
- Fixed costs: f = [97.65, 99.76, 100.76, 105.32, 98.88]
- Demands: d = [143, 6, 10, 25, 3]
- Transportation cost matrix c (rows: S1–S5, columns: C1–C5):

\[
c = \begin{bmatrix}
150.74 & 0.02 & 49.13 & 2080.15 & 426.40 \\
233.05 & 97.73 & 49.84 & 1982.39 & 23.96 \\
55.68 & 935.61 & 4.03 & 73.09 & 525.32 \\
1483.82 & 1801.08 & 112.16 & 816.05 & 107.01 \\
1119.47 & 884.31 & 0.08 & 1544.95 & 543.67 \\
\end{bmatrix}
\]

Summary:
The model determines which suppliers to open (y_i) and how much each branch sources from each supplier (x_{ij}) to minimize the sum of fixed and transportation costs, while meeting all branch demands and only sourcing from open suppliers. All parameters (vectors and matrices) are explicitly listed above.