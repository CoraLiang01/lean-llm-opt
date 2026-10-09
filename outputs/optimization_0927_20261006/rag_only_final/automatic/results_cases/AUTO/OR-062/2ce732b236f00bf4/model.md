Let us define the following sets and parameters based on the CSV data:

Sets:
- Let F = {MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES} be the set of suppliers (indexed by i).
- Let S = {CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT} be the set of stores (indexed by j).
- Let C = {Customer_1, Customer_2, Customer_3, Customer_4, Customer_5} be the set of demand points (indexed by k).

Parameters:
- Fixed costs for each supplier:
    - f = [96.58, 94.06, 94.37, 82.88, 94.96] corresponding to F in the order above.
- Demand for each store (assuming mapping Customer_k to store j in order):
    - d = [2397, 1889, 2518, 3218, 1813] corresponding to S in the order above.
- Transportation costs per unit from each supplier to each store (matrix c_{ij}):
    - c =

|                | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|----------------|----------|--------------|------------|--------|----------|
| MOUNT AYR      | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE         | 15.13    | 1.50         | 1.43       | 27.88  | 90.69    |
| WAVERLY        | 2.34     | 349.34       | 246.60     | 41.30  | 78.73    |
| PELLA          | 1181.60  | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES     | 1030.80  | 43.48        | 932.43     | 55.39  | 103.84   |

Decision Variables:
- y_i ∈ {0,1} for each supplier i ∈ F, where y_i = 1 if supplier i is activated (open), 0 otherwise.
- x_{ij} ≥ 0: quantity of goods supplied from supplier i ∈ F to store j ∈ S.

Mathematical Model:

Objective:
Minimize the total cost, which is the sum of fixed costs for activated suppliers and the total transportation cost:
\[
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij}
\]
where:
- \( f_i \) is the fixed cost for supplier i,
- \( c_{ij} \) is the transportation cost per unit from supplier i to store j,
- \( y_i \) is the binary variable indicating if supplier i is open,
- \( x_{ij} \) is the quantity supplied from i to j.

Subject to:

1. Demand satisfaction at each store:
\[
\sum_{i \in F} x_{ij} = d_j \quad \forall j \in S
\]
where \( d_j \) is the demand at store j.

2. Supplier activation constraint:
\[
x_{ij} \leq d_j y_i \quad \forall i \in F, \forall j \in S
\]
(This ensures that if supplier i is not open, it cannot supply any goods.)

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in F
\]
\[
x_{ij} \geq 0 \quad \forall i \in F, \forall j \in S
\]

Explicit parameter values:

- F = {MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES}
- S = {CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT}
- f = [96.58, 94.06, 94.37, 82.88, 94.96]
- d = [2397, 1889, 2518, 3218, 1813]
- c =

\[
\begin{bmatrix}
694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
15.13 & 1.50 & 1.43 & 27.88 & 90.69 \\
2.34 & 349.34 & 246.60 & 41.30 & 78.73 \\
1181.60 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
1030.80 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{bmatrix}
\]

where rows correspond to F in the order above and columns to S in the order above.

Summary:
The mathematical model is:

\[
\min \left( 96.58 y_1 + 94.06 y_2 + 94.37 y_3 + 82.88 y_4 + 94.96 y_5 + \sum_{i=1}^5 \sum_{j=1}^5 c_{ij} x_{ij} \right)
\]
subject to
\[
\sum_{i=1}^5 x_{ij} = d_j \quad \forall j=1,\ldots,5
\]
\[
x_{ij} \leq d_j y_i \quad \forall i=1,\ldots,5; \; j=1,\ldots,5
\]
\[
y_i \in \{0,1\} \quad \forall i=1,\ldots,5
\]
\[
x_{ij} \geq 0 \quad \forall i=1,\ldots,5; \; j=1,\ldots,5
\]

with all parameter values as specified above.