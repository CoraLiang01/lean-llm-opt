Let:
- Suppliers: S = {S1, S2, S3, S4, S5}
- Branches (customers): C = {C1, C2, C3, C4, C5}

Parameters:
- Fixed costs for opening supplier i (f_i):
  - f_S1 = 97.65
  - f_S2 = 99.76
  - f_S3 = 100.76
  - f_S4 = 105.32
  - f_S5 = 98.88

- Transportation costs per unit from supplier i to branch j (c_{ij}):

|        | C1      | C2      | C3    | C4      | C5      |
|--------|---------|---------|-------|---------|---------|
| S1     | 150.74  | 0.02    | 49.13 | 2080.15 | 426.40  |
| S2     | 233.05  | 97.73   | 49.84 | 1982.39 | 23.96   |
| S3     | 55.68   | 935.61  | 4.03  | 73.09   | 525.32  |
| S4     | 1483.82 | 1801.08 | 112.16| 816.05  | 107.01  |
| S5     | 1119.47 | 884.31  | 0.08  | 1544.95 | 543.67  |

- Demand at each branch j (d_j):
  - d_C1 = 143
  - d_C2 = 6
  - d_C3 = 10
  - d_C4 = 25
  - d_C5 = 3

Decision variables:
- y_i ∈ {0,1} for i ∈ S: 1 if supplier i is open, 0 otherwise.
- x_{ij} ≥ 0 for i ∈ S, j ∈ C: quantity supplied from supplier i to branch j.

Mathematical Model:

Minimize total cost:
\[
\text{Minimize} \quad Z = \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij}
\]
where:
- \( f_i \) is the fixed cost for supplier i,
- \( c_{ij} \) is the transportation cost per unit from supplier i to branch j,
- \( x_{ij} \) is the quantity supplied from i to j,
- \( y_i \) is the binary variable indicating if supplier i is open.

Subject to:
1. Demand satisfaction at each branch:
   \[
   \sum_{i \in S} x_{ij} = d_j \quad \forall j \in C
   \]
2. Supply only from open suppliers:
   \[
   x_{ij} \leq d_j y_i \quad \forall i \in S, \forall j \in C
   \]
   (This ensures that if y_i = 0, then x_{ij} = 0 for all j.)

3. Variable domains:
   \[
   y_i \in \{0,1\} \quad \forall i \in S
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in S, \forall j \in C
   \]

All parameters (vectors and matrices):

- f = [97.65, 99.76, 100.76, 105.32, 98.88]
- c = 
\[
\begin{bmatrix}
150.74 & 0.02 & 49.13 & 2080.15 & 426.40 \\
233.05 & 97.73 & 49.84 & 1982.39 & 23.96 \\
55.68 & 935.61 & 4.03 & 73.09 & 525.32 \\
1483.82 & 1801.08 & 112.16 & 816.05 & 107.01 \\
1119.47 & 884.31 & 0.08 & 1544.95 & 543.67 \\
\end{bmatrix}
\]
- d = [143, 6, 10, 25, 3]

Summary:
This is a capacitated facility location problem (uncapacitated in this case, as no supplier capacity is given), with binary supplier activation variables and continuous allocation variables, minimizing the sum of fixed and transportation costs, subject to demand satisfaction and supply only from open suppliers.