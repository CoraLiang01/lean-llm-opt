Let:
- I = {S1, S2, S3, S4, S5} be the set of suppliers.
- J = {C1, C2, C3, C4, C5} be the set of branches.

Parameters:
- Fixed costs for each supplier:
  - f_S1 = 97.65
  - f_S2 = 99.76
  - f_S3 = 100.76
  - f_S4 = 105.32
  - f_S5 = 98.88

- Demand for each branch:
  - d_C1 = 143
  - d_C2 = 6
  - d_C3 = 10
  - d_C4 = 25
  - d_C5 = 3

- Transportation costs c_{ij} (cost per unit from supplier i to branch j):

|        | C1      | C2      | C3      | C4      | C5      |
|--------|---------|---------|---------|---------|---------|
| S1     | 150.74  | 0.02    | 49.13   | 2080.15 | 426.40  |
| S2     | 233.05  | 97.73   | 49.84   | 1982.39 | 23.96   |
| S3     | 55.68   | 935.61  | 4.03    | 73.09   | 525.32  |
| S4     | 1483.82 | 1801.08 | 112.16  | 816.05  | 107.01  |
| S5     | 1119.47 | 884.31  | 0.08    | 1544.95 | 543.67  |

Decision variables:
- y_i ∈ {0,1} for each supplier i ∈ I, where y_i = 1 if supplier i is operational, 0 otherwise.
- x_{ij} ≥ 0: quantity of goods supplied from supplier i ∈ I to branch j ∈ J.

Mathematical Model:

Objective:
Minimize total cost (fixed + transportation):
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
That is,
\[
\min \Big[
97.65\,y_{S1} + 99.76\,y_{S2} + 100.76\,y_{S3} + 105.32\,y_{S4} + 98.88\,y_{S5}
+ 150.74\,x_{S1,C1} + 0.02\,x_{S1,C2} + 49.13\,x_{S1,C3} + 2080.15\,x_{S1,C4} + 426.40\,x_{S1,C5}
+ 233.05\,x_{S2,C1} + 97.73\,x_{S2,C2} + 49.84\,x_{S2,C3} + 1982.39\,x_{S2,C4} + 23.96\,x_{S2,C5}
+ 55.68\,x_{S3,C1} + 935.61\,x_{S3,C2} + 4.03\,x_{S3,C3} + 73.09\,x_{S3,C4} + 525.32\,x_{S3,C5}
+ 1483.82\,x_{S4,C1} + 1801.08\,x_{S4,C2} + 112.16\,x_{S4,C3} + 816.05\,x_{S4,C4} + 107.01\,x_{S4,C5}
+ 1119.47\,x_{S5,C1} + 884.31\,x_{S5,C2} + 0.08\,x_{S5,C3} + 1544.95\,x_{S5,C4} + 543.67\,x_{S5,C5}
\Big]
\]

Subject to:

1. Demand satisfaction for each branch:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]
That is,
\[
x_{S1,C1} + x_{S2,C1} + x_{S3,C1} + x_{S4,C1} + x_{S5,C1} = 143
\]
\[
x_{S1,C2} + x_{S2,C2} + x_{S3,C2} + x_{S4,C2} + x_{S5,C2} = 6
\]
\[
x_{S1,C3} + x_{S2,C3} + x_{S3,C3} + x_{S4,C3} + x_{S5,C3} = 10
\]
\[
x_{S1,C4} + x_{S2,C4} + x_{S3,C4} + x_{S4,C4} + x_{S5,C4} = 25
\]
\[
x_{S1,C5} + x_{S2,C5} + x_{S3,C5} + x_{S4,C5} + x_{S5,C5} = 3
\]

2. Linking constraints: Only allow supply from open suppliers:
\[
x_{ij} \leq d_j y_i \quad \forall i \in I,\, j \in J
\]

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

All parameters, vectors, and matrices are as specified above. This model determines which suppliers to activate and how to allocate supply to branches to minimize total cost while meeting all demand.