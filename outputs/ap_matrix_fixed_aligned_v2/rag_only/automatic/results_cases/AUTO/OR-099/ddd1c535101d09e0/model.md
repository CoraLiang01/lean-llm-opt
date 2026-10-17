Let:
- I = {1,2,...,11} be the set of potential warehouses.
- J = {1,2,...,11} be the set of stores.

Parameters:
- fi: Opening cost for warehouse i.
  f = [3000, 3200, 3100, 2800, 3500, 2700, 2900, 3050, 3100, 2200, 2890]
  (where f[1] = 3000 for warehouse 1, etc.)
- ci,j: Transportation cost per unit from warehouse i to store j.
  The cost matrix C = [c_ij] is:

|      |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |  9 | 10 | 11 |
|------|----|----|----|----|----|----|----|----|----|----|----|
|  1   | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14 | 15 |
|  2   | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15 | 16 |
|  3   | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18 | 17 |
|  4   | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13 | 18 |
|  5   | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15 | 17 |
|  6   | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18 | 19 |
|  7   | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16 | 14 |
|  8   | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15 | 18 |
|  9   | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15 | 18 |
| 10   | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17 | 19 |
| 11   | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21 | 13 |

- Capacity of warehouse i: cap = [180, 160, 200, 150, 170, 190, 160, 175, 170, 180, 190]
- Demand of store j: d = [30, 40, 20, 35, 20, 25, 45, 38, 32, 41, 44]

Decision variables:
- yi ∈ {0,1}: 1 if warehouse i is opened, 0 otherwise.
- x_ij ≥ 0: Amount supplied from warehouse i to store j.

Mathematical Model:

Minimize total cost:
\[
\text{Minimize} \quad Z = \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij}
\]
where:
- \( f_i \) is the opening cost for warehouse i (see vector above)
- \( c_{ij} \) is the transportation cost from warehouse i to store j (see matrix above)
- \( y_i \) is the binary open/close variable for warehouse i
- \( x_{ij} \) is the amount shipped from warehouse i to store j

Subject to:

1. Demand satisfaction for each store:
\[
\sum_{i=1}^{11} x_{ij} = d_j \quad \forall j = 1,\ldots,11
\]
where \( d_j \) is the demand for store j (see vector above).

2. Warehouse capacity:
\[
\sum_{j=1}^{11} x_{ij} \leq \text{cap}_i \cdot y_i \quad \forall i = 1,\ldots,11
\]
where \( \text{cap}_i \) is the capacity of warehouse i (see vector above).

3. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i, j
\]

4. Binary warehouse open/close:
\[
y_i \in \{0,1\} \quad \forall i
\]

All parameters (vectors and matrices) are as specified above, directly from the CSV data. This is a standard uncapacitated facility location problem with capacities and assignment.