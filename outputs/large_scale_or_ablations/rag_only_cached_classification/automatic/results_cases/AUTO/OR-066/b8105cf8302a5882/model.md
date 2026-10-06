Let:
- I = {S1, S2} be the set of suppliers (facilities)
- J = {C1, C2} be the set of supermarkets (customers)

Parameters:
- Fixed cost vector for suppliers:
  f = [f_S1, f_S2] = [105.97, 85.31]
- Demand vector for supermarkets:
  d = [d_C1, d_C2] = [144, 216]
- Transportation cost matrix (c_{ij}), where c_{ij} is the per-unit transportation cost from supplier i to supermarket j:

|        | C1      | C2     |
|--------|---------|--------|
| S1     | 2358.39 | 1492.08|
| S2     | 0.07    | 52.32  |

Variables:
- y_i ∈ {0,1} for each supplier i ∈ I, where y_i = 1 if supplier i is activated, 0 otherwise
- x_{ij} ≥ 0 for each supplier i ∈ I and supermarket j ∈ J, representing the amount supplied from i to j

Mathematical Model:

Minimize total cost:
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
Explicitly, with the data:
\[
\min \left(105.97\, y_{S1} + 85.31\, y_{S2}\right) + \left(2358.39\, x_{S1,C1} + 1492.08\, x_{S1,C2} + 0.07\, x_{S2,C1} + 52.32\, x_{S2,C2}\right)
\]

Subject to:
1. Demand satisfaction for each supermarket:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]
Explicitly:
\[
x_{S1,C1} + x_{S2,C1} = 144
\]
\[
x_{S1,C2} + x_{S2,C2} = 216
\]

2. Supply only from open suppliers:
\[
x_{ij} \leq d_j y_i \quad \forall i \in I,\, j \in J
\]
Explicitly:
\[
x_{S1,C1} \leq 144\, y_{S1}
\]
\[
x_{S1,C2} \leq 216\, y_{S1}
\]
\[
x_{S2,C1} \leq 144\, y_{S2}
\]
\[
x_{S2,C2} \leq 216\, y_{S2}
\]

3. Binary and non-negativity constraints:
\[
y_i \in \{0,1\} \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

Summary of parameters:
- Suppliers: I = {S1, S2}
- Supermarkets: J = {C1, C2}
- Fixed costs: f = [105.97, 85.31]
- Demands: d = [144, 216]
- Transportation cost matrix:
  c = \(\begin{bmatrix} 2358.39 & 1492.08 \\ 0.07 & 52.32 \end{bmatrix}\)
  (rows: S1, S2; columns: C1, C2)

This is the full mathematical model for the supplier-supermarket replenishment problem, with all required parameters, vectors, and matrices extracted from the provided CSV data.