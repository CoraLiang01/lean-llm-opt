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

Decision Variables:
- y_{S1}, y_{S2} ∈ {0,1}
- x_{S1,C1}, x_{S1,C2}, x_{S2,C1}, x_{S2,C2} ≥ 0

Objective:
Minimize total cost:
\[
\text{Minimize} \quad 
105.97\,y_{S1} + 85.31\,y_{S2} 
+ 2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} 
+ 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2}
\]

Subject to:

1. Demand satisfaction for each supermarket:
\[
x_{S1,C1} + x_{S2,C1} = 144
\]
\[
x_{S1,C2} + x_{S2,C2} = 216
\]

2. Linking constraints (only supply from open suppliers):
\[
x_{S1,C1} + x_{S1,C2} \leq (144 + 216)\,y_{S1} = 360\,y_{S1}
\]
\[
x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2}
\]

3. Variable domains:
\[
y_{S1}, y_{S2} \in \{0,1\}
\]
\[
x_{S1,C1}, x_{S1,C2}, x_{S2,C1}, x_{S2,C2} \geq 0
\]

Where:
- y_{S1}, y_{S2}: 1 if supplier S1 or S2 is activated, 0 otherwise.
- x_{ij}: units supplied from supplier i to supermarket j.

All parameters (fixed costs, demands, transportation costs) are as retrieved from the CSV files. This model minimizes the sum of fixed and transportation costs while ensuring all supermarket demands are met and only activated suppliers can supply goods.