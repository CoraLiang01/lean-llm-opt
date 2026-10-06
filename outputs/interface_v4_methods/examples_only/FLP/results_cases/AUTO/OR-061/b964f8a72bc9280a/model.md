Let:
- I = {1, 2, 3, 4, 5} be the set of suppliers, corresponding to S1, S2, S3, S4, S5.
- J = {1, 2, 3, 4, 5} be the set of branches/customers, corresponding to C1, C2, C3, C4, C5.

Parameters:
- Fixed costs for each supplier:
  f = [97.65, 99.76, 100.76, 105.32, 98.88], where f_i is the fixed cost for supplier S_i.

- Demand for each branch:
  d = [143, 6, 10, 25, 3], where d_j is the demand for branch C_j.

- Transportation cost matrix (c_{ij}), where c_{ij} is the cost per unit from supplier S_i to branch C_j:

  c = 
  |      C1    C2    C3    C4    C5  |
  | S1 | 150.74 0.02 49.13 2080.15 426.40 |
  | S2 | 233.05 97.73 49.84 1982.39 23.96 |
  | S3 | 55.68 935.61 4.03 73.09 525.32 |
  | S4 | 1483.82 1801.08 112.16 816.05 107.01 |
  | S5 | 1119.47 884.31 0.08 1544.95 543.67 |

Decision variables:
- y_i ∈ {0,1}, for i ∈ I: y_i = 1 if supplier S_i is operational (open), 0 otherwise.
- x_{ij} ≥ 0, for i ∈ I, j ∈ J: quantity of goods supplied from supplier S_i to branch C_j.

Mathematical Model:

Minimize total cost:
\[
\text{Minimize} \quad Z = \sum_{i=1}^{5} f_i y_i + \sum_{i=1}^{5} \sum_{j=1}^{5} c_{ij} x_{ij}
\]
where:
- f = [97.65, 99.76, 100.76, 105.32, 98.88]
- c_{ij} as given above

Subject to:
1. Demand satisfaction for each branch:
\[
\sum_{i=1}^{5} x_{ij} = d_j \quad \forall j \in \{1,2,3,4,5\}
\]
where d = [143, 6, 10, 25, 3]

2. Supply only from open suppliers:
\[
x_{ij} \leq d_j y_i \quad \forall i \in \{1,2,3,4,5\}, \forall j \in \{1,2,3,4,5\}
\]

3. Binary and non-negativity constraints:
\[
y_i \in \{0,1\} \quad \forall i \in \{1,2,3,4,5\}
\]
\[
x_{ij} \geq 0 \quad \forall i \in \{1,2,3,4,5\}, \forall j \in \{1,2,3,4,5\}
\]

Where:
- i indexes suppliers: S1, S2, S3, S4, S5
- j indexes branches: C1, C2, C3, C4, C5

All vectors and matrices are as explicitly stated above. This model determines which suppliers to open and how to allocate supply to branches to minimize the sum of fixed and transportation costs, while meeting all branch demands.