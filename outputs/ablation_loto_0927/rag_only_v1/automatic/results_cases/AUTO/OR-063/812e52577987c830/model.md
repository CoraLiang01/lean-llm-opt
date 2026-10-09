Mathematical Model for Bandcamp Inventory Replenishment via Warehouses

Sets:
- Let F = {S1, S2, S3, S4, S5, S6, S7} be the set of warehouses (facilities).
- Let C = {C1, C2, C3, C4, C5, C6, C7} be the set of musicians/bands (customers).

Parameters:
- Fixed cost for opening warehouse i ∈ F: 
  - f = [f_S1, f_S2, f_S3, f_S4, f_S5, f_S6, f_S7] = [102.33, 94.92, 91.83, 98.71, 95.73, 99.96, 98.16]
- Transportation cost per unit from warehouse i ∈ F to customer j ∈ C: 
  - c_{ij} matrix (rows: S1–S7, columns: C1–C7):

|        | C1      | C2      | C3      | C4      | C5      | C6      | C7      |
|--------|---------|---------|---------|---------|---------|---------|---------|
| S1     | 1506.22 | 70.90   | 8.44    | 260.27  | 197.47  | 71.71   | 61.19   |
| S2     | 1732.65 | 1780.72 | 567.44  | 448.68  | 29.00   | 1484.91 | 963.92  |
| S3     | 115.66  | 100.76  | 64.68   | 1324.53 | 64.99   | 134.88  | 2102.83 |
| S4     | 1254.78 | 1115.63 | 52.31   | 1036.16 | 892.63  | 1464.04 | 1383.41 |
| S5     | 42.90   | 891.01  | 1013.94 | 1128.72 | 58.91   | 42.89   | 1570.31 |
| S6     | 0.70    | 139.46  | 70.03   | 79.15   | 1482.00 | 0.91    | 110.46  |
| S7     | 1732.30 | 1780.44 | 486.50  | 523.74  | 522.08  | 82.48   | 826.41  |

- Demand for each customer j ∈ C:
  - d = [d_C1, d_C2, d_C3, d_C4, d_C5, d_C6, d_C7] = [1083, 776, 16214, 553, 17106, 594, 732]

Decision Variables:
- y_i ∈ {0,1} for each i ∈ F: 1 if warehouse i is opened, 0 otherwise.
- x_{ij} ≥ 0 for each i ∈ F, j ∈ C: quantity of goods supplied from warehouse i to customer j.

Objective:
Minimize the total cost, which is the sum of fixed warehouse opening costs and transportation costs:
\[
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} c_{ij} x_{ij}
\]
where:
- \( f_i \) is the fixed cost for warehouse i (see vector above),
- \( c_{ij} \) is the transportation cost per unit from warehouse i to customer j (see matrix above),
- \( x_{ij} \) is the quantity supplied from i to j,
- \( y_i \) is the binary variable indicating if warehouse i is open.

Constraints:
1. Demand satisfaction for each customer:
   \[
   \sum_{i \in F} x_{ij} = d_j \quad \forall j \in C
   \]
   where \( d_j \) is the demand for customer j (see vector above).

2. Linking constraint: Only supply from open warehouses:
   \[
   x_{ij} \leq d_j y_i \quad \forall i \in F, \forall j \in C
   \]

3. Binary and non-negativity:
   \[
   y_i \in \{0,1\} \quad \forall i \in F
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in F, \forall j \in C
   \]

All parameters (fixed costs, transportation costs, and demands) are explicitly stated above. This model determines which warehouses to open and how to allocate supply to minimize total cost while meeting all musician/band demands.