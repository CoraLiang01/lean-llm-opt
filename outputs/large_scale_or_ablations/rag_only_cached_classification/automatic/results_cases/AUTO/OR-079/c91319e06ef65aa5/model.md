Let:
- I = {A1, A2, ..., A15} be the set of potential factory sites (indexed by i)
- J = {B1, B2, ..., B8} be the set of distribution centers (indexed by j)

Parameters (from CSVs):

Factory fixed costs and capacities:
- f = [0, 175, 300, 375, 500, 200, 260, 220, 320, 280, 350, 420, 470, 520, 560] // f[i] is the fixed cost for factory Ai
- cap = [30, 10, 20, 30, 40, 20, 25, 30, 35, 20, 40, 25, 30, 50, 45] // cap[i] is the capacity for factory Ai

Distribution center demands:
- d = [30, 25, 20, 35, 25, 30, 25, 30] // d[j] is the demand for distribution center Bj

Shipping costs (matrix c[i][j], i=1..15, j=1..8):

c = [
  [8, 4, 3, 6, 7, 5, 9, 8],    // A1
  [5, 2, 3, 5, 6, 4, 7, 6],    // A2
  [4, 3, 4, 6, 5, 5, 6, 7],    // A3
  [9, 7, 5, 8, 9, 6, 10, 7],   // A4
  [10, 4, 2, 6, 8, 5, 7, 3],   // A5
  [6, 5, 4, 5, 7, 6, 8, 5],    // A6
  [7, 6, 5, 4, 6, 7, 9, 6],    // A7
  [5, 4, 6, 3, 5, 6, 7, 6],    // A8
  [8, 7, 6, 7, 9, 8, 10, 7],   // A9
  [6, 5, 7, 4, 6, 5, 7, 5],    // A10
  [9, 6, 4, 6, 8, 7, 9, 6],    // A11
  [7, 5, 6, 5, 6, 5, 8, 5],    // A12
  [8, 6, 5, 6, 7, 6, 8, 7],    // A13
  [9, 5, 3, 5, 7, 4, 6, 4],    // A14
  [10, 6, 4, 5, 8, 5, 7, 5]    // A15
]

Decision variables:
- y_i ∈ {0,1} for i ∈ I // y_i = 1 if factory i is built, 0 otherwise
- x_{ij} ≥ 0 for i ∈ I, j ∈ J // units shipped from factory i to distribution center j

Mathematical Model:

Minimize total system cost:
\[
\text{Minimize} \quad Z = \sum_{i=1}^{15} f_i y_i + \sum_{i=1}^{15} \sum_{j=1}^{8} c_{ij} x_{ij}
\]
where:
- \( f_i \) is the fixed cost for factory i (see vector above)
- \( c_{ij} \) is the shipping cost per unit from factory i to distribution center j (see matrix above)

Subject to:

1. Demand satisfaction at each distribution center:
\[
\sum_{i=1}^{15} x_{ij} = d_j \quad \forall j = 1,\ldots,8
\]
where \( d_j \) is the demand for distribution center j (see vector above)

2. Factory capacity and open/close logic:
\[
\sum_{j=1}^{8} x_{ij} \leq cap_i \cdot y_i \quad \forall i = 1,\ldots,15
\]
where \( cap_i \) is the capacity for factory i (see vector above)

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i = 1,\ldots,15
\]
\[
x_{ij} \geq 0 \quad \forall i = 1,\ldots,15; \; j = 1,\ldots,8
\]

All parameters (fixed costs, capacities, demands, shipping costs) are explicitly provided above.

This is a classic capacitated facility location problem (CFLP) model for ElectroTech Manufacturing’s network restructuring, preserving the original objective and all constant terms.