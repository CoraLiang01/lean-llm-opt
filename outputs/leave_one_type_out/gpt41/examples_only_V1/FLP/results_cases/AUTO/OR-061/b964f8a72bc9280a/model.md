Let:
- I = {S1, S2, S3, S4, S5} be the set of suppliers.
- J = {C1, C2, C3, C4, C5} be the set of branches.

Parameters:
- Fixed cost for each supplier:
    - f = [f_S1, f_S2, f_S3, f_S4, f_S5] = [97.65, 99.76, 100.76, 105.32, 98.88]
      where f_S1 = 97.65, f_S2 = 99.76, f_S3 = 100.76, f_S4 = 105.32, f_S5 = 98.88
- Demand for each branch:
    - d = [d_C1, d_C2, d_C3, d_C4, d_C5] = [143, 6, 10, 25, 3]
      where d_C1 = 143, d_C2 = 6, d_C3 = 10, d_C4 = 25, d_C5 = 3
- Transportation cost per unit from supplier i to branch j:
    - c = [ [150.74, 0.02, 49.13, 2080.15, 426.4],
            [233.05, 97.73, 49.84, 1982.39, 23.96],
            [55.68, 935.61, 4.03, 73.09, 525.32],
            [1483.82, 1801.08, 112.16, 816.05, 107.01],
            [1119.47, 884.31, 0.08, 1544.95, 543.67] ]
      where c_{ij} is the cost from supplier S_i to branch C_j.

Decision Variables:
- y_i ∈ {0,1} for i ∈ I: y_i = 1 if supplier i is operational (open), 0 otherwise.
- x_{ij} ≥ 0 for i ∈ I, j ∈ J: quantity of goods supplied from supplier i to branch j.

Mathematical Model:

Objective:
Minimize total cost (fixed + transportation):
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
That is,
\[
\min \left(
97.65 y_{S1} + 99.76 y_{S2} + 100.76 y_{S3} + 105.32 y_{S4} + 98.88 y_{S5}
+ 150.74 x_{S1,C1} + 0.02 x_{S1,C2} + 49.13 x_{S1,C3} + 2080.15 x_{S1,C4} + 426.4 x_{S1,C5}
+ 233.05 x_{S2,C1} + 97.73 x_{S2,C2} + 49.84 x_{S2,C3} + 1982.39 x_{S2,C4} + 23.96 x_{S2,C5}
+ 55.68 x_{S3,C1} + 935.61 x_{S3,C2} + 4.03 x_{S3,C3} + 73.09 x_{S3,C4} + 525.32 x_{S3,C5}
+ 1483.82 x_{S4,C1} + 1801.08 x_{S4,C2} + 112.16 x_{S4,C3} + 816.05 x_{S4,C4} + 107.01 x_{S4,C5}
+ 1119.47 x_{S5,C1} + 884.31 x_{S5,C2} + 0.08 x_{S5,C3} + 1544.95 x_{S5,C4} + 543.67 x_{S5,C5}
\right)
\]

Subject to:

1. Demand satisfaction at each branch:
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

2. Linking supplier activation to supply:
\[
x_{ij} \leq d_j y_i \quad \forall i \in I, j \in J
\]
(If supplier i is not open, it cannot supply any goods.)

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I, j \in J
\]

All parameters (fixed costs, transportation costs, demands) are explicitly listed above. This model determines which suppliers to open and how much each branch should source from each supplier to minimize the total cost while meeting all demands.