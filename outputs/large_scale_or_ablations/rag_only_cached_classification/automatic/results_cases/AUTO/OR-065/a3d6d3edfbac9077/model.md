Let:
- F = {S1, S2, S3} be the set of warehouses (facilities).
- C = {C1, C2, C3} be the set of musicians/bands (customers).

Parameters:
- Fixed cost vector for warehouses:
  - f = [f1, f2, f3] = [102.33, 94.92, 91.83], where fi is the fixed cost of warehouse Si.
- Demand vector for customers:
  - d = [d1, d2, d3] = [1083, 776, 16214], where dj is the demand of customer Cj.
- Transportation cost matrix (per unit from warehouse Si to customer Cj):
  - c = [ [1506.22, 70.90, 8.44],
           [1732.65, 1780.72, 567.44],
           [115.66, 100.76, 64.68] ]

Decision variables:
- y_i ∈ {0,1} for i ∈ {1,2,3}: y_i = 1 if warehouse Si is opened, 0 otherwise.
- x_{ij} ≥ 0 for i ∈ {1,2,3}, j ∈ {1,2,3}: x_{ij} is the quantity of goods supplied from warehouse Si to customer Cj.

Mathematical Model:

Minimize total cost:
\[
\text{Minimize} \quad Z = \sum_{i=1}^{3} f_i y_i + \sum_{i=1}^{3} \sum_{j=1}^{3} c_{ij} x_{ij}
\]
where:
- f_1 = 102.33, f_2 = 94.92, f_3 = 91.83
- c_{ij} is the (i,j)th entry of the transportation cost matrix above

Subject to:
1. Demand satisfaction for each customer:
\[
\sum_{i=1}^{3} x_{ij} = d_j \quad \forall j \in \{1,2,3\}
\]
where d_1 = 1083, d_2 = 776, d_3 = 16214

2. Supply only from open warehouses:
\[
x_{ij} \leq d_j y_i \quad \forall i \in \{1,2,3\}, \forall j \in \{1,2,3\}
\]

3. Binary and non-negativity constraints:
\[
y_i \in \{0,1\} \quad \forall i \in \{1,2,3\}
\]
\[
x_{ij} \geq 0 \quad \forall i \in \{1,2,3\}, \forall j \in \{1,2,3\}
\]

Where:
- y_i: binary variable indicating if warehouse Si is open
- x_{ij}: quantity supplied from warehouse Si to customer Cj

All required parameters, vectors, and matrices are as follows:

Warehouses (F): S1, S2, S3  
Musicians/Bands (C): C1, C2, C3  
Fixed cost vector (f): [102.33, 94.92, 91.83]  
Demand vector (d): [1083, 776, 16214]  
Transportation cost matrix (c):

|        | C1      | C2      | C3     |
|--------|---------|---------|--------|
| S1     | 1506.22 | 70.90   | 8.44   |
| S2     | 1732.65 | 1780.72 | 567.44 |
| S3     | 115.66  | 100.76  | 64.68  |

Decision variables:  
y_i ∈ {0,1} for i = 1,2,3  
x_{ij} ≥ 0 for i = 1,2,3; j = 1,2,3

Objective and constraints as above.