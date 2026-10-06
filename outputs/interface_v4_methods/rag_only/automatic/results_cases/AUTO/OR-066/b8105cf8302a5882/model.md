Let:
- F = {S1, S2} be the set of suppliers (facilities).
- C = {C1, C2} be the set of supermarkets (customers).
- y_i ∈ {0,1} for i ∈ F: binary variable indicating if supplier i is activated.
- x_{ij} ≥ 0 for i ∈ F, j ∈ C: amount supplied from supplier i to supermarket j.

Parameters:
- Fixed costs:
    - f_{S1} = 105.97
    - f_{S2} = 85.31
- Demands:
    - d_{C1} = 144
    - d_{C2} = 216
- Transportation costs (per unit):
    - c_{S1,C1} = 2358.39
    - c_{S1,C2} = 1492.08
    - c_{S2,C1} = 0.07
    - c_{S2,C2} = 52.32

Mathematical Model:

Decision variables:
- y_{S1}, y_{S2} ∈ {0,1}
- x_{S1,C1}, x_{S1,C2}, x_{S2,C1}, x_{S2,C2} ≥ 0

Objective:
Minimize total cost:
\[
\min \left[
    105.97\, y_{S1} + 85.31\, y_{S2}
    + 2358.39\, x_{S1,C1} + 1492.08\, x_{S1,C2}
    + 0.07\, x_{S2,C1} + 52.32\, x_{S2,C2}
\right]
\]

Subject to:
1. Demand satisfaction for each supermarket:
    - x_{S1,C1} + x_{S2,C1} = 144
    - x_{S1,C2} + x_{S2,C2} = 216

2. Linking constraints (only supply from open suppliers):
    - x_{S1,C1} + x_{S1,C2} ≤ (144 + 216) y_{S1} = 360 y_{S1}
    - x_{S2,C1} + x_{S2,C2} ≤ 360 y_{S2}

3. Variable domains:
    - y_{S1}, y_{S2} ∈ {0,1}
    - x_{S1,C1}, x_{S1,C2}, x_{S2,C1}, x_{S2,C2} ≥ 0

Summary of parameters:
- Suppliers: F = [S1, S2]
- Supermarkets: C = [C1, C2]
- Fixed cost vector: f = [105.97, 85.31]
- Demand vector: d = [144, 216]
- Transportation cost matrix:
    \[
    \begin{bmatrix}
    2358.39 & 1492.08 \\
    0.07 & 52.32
    \end{bmatrix}
    \]
    (Rows: S1, S2; Columns: C1, C2)

This model determines which suppliers to activate (y_i) and how much each supplier should deliver to each supermarket (x_{ij}) to minimize the sum of fixed and transportation costs, while satisfying all supermarket demands.