Let:
- I = {1, 2, 3, 4, 5} be the set of suppliers, indexed by i (corresponding to S1–S5)
- J = {1, 2, 3, 4, 5} be the set of branches, indexed by j (corresponding to C1–C5)

Parameters:
- Fixed costs for suppliers (vector f):  
  f = [97.65, 99.76, 100.76, 105.32, 98.88]  
  where f_i is the fixed cost for supplier S_i.

- Demand for each branch (vector d):  
  d = [143, 6, 10, 25, 3]  
  where d_j is the demand for branch C_j.

- Transportation cost matrix (C):  
  C =  
  |         | C1     | C2     | C3    | C4      | C5     |
  |---------|--------|--------|-------|---------|--------|
  | S1      | 150.74 | 0.02   | 49.13 | 2080.15 | 426.40 |
  | S2      | 233.05 | 97.73  | 49.84 | 1982.39 | 23.96  |
  | S3      | 55.68  | 935.61 | 4.03  | 73.09   | 525.32 |
  | S4      | 1483.82| 1801.08| 112.16| 816.05  | 107.01 |
  | S5      | 1119.47| 884.31 | 0.08  | 1544.95 | 543.67 |

Decision variables:
- y_i ∈ {0,1} for i ∈ I, where y_i = 1 if supplier S_i is operational, 0 otherwise.
- x_{ij} ≥ 0 for i ∈ I, j ∈ J, where x_{ij} is the quantity of goods supplied from supplier S_i to branch C_j.

Mathematical Model:

Minimize total cost:
\[
\text{Minimize} \quad Z = \sum_{i=1}^5 f_i y_i + \sum_{i=1}^5 \sum_{j=1}^5 C_{ij} x_{ij}
\]
where:
- \( f = [97.65, 99.76, 100.76, 105.32, 98.88] \)
- \( C_{ij} \) is as given in the matrix above.

Subject to:
1. Demand satisfaction for each branch:
\[
\sum_{i=1}^5 x_{ij} = d_j \quad \forall j \in \{1,2,3,4,5\}
\]
where \( d = [143, 6, 10, 25, 3] \).

2. Supply only from open suppliers:
\[
x_{ij} \leq d_j y_i \quad \forall i \in \{1,2,3,4,5\},\; \forall j \in \{1,2,3,4,5\}
\]

3. Binary and non-negativity constraints:
\[
y_i \in \{0,1\} \quad \forall i \in \{1,2,3,4,5\}
\]
\[
x_{ij} \geq 0 \quad \forall i \in \{1,2,3,4,5\},\; \forall j \in \{1,2,3,4,5\}
\]

Where all parameters (fixed costs, demands, and transportation costs) are as explicitly listed above. This model determines which suppliers to activate and how to allocate supply to branches to minimize the total cost.