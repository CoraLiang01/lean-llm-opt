Let:
- There are 2 suppliers: S1 and S2.
- There are 2 supermarkets (customers): C1 and C2.

Parameters:
- Fixed costs for activating each supplier:
    - f₁ = 105.97 (for S1)
    - f₂ = 85.31 (for S2)
- Transportation costs per unit from each supplier to each supermarket:
    - c₁₁ = 2358.39 (S1 to C1)
    - c₁₂ = 1492.08 (S1 to C2)
    - c₂₁ = 0.07    (S2 to C1)
    - c₂₂ = 52.32   (S2 to C2)
- Demand at each supermarket:
    - d₁ = 144 (C1)
    - d₂ = 216 (C2)

Decision Variables:
- y₁ ∈ {0,1}: 1 if supplier S1 is activated, 0 otherwise
- y₂ ∈ {0,1}: 1 if supplier S2 is activated, 0 otherwise
- x₁₁ ≥ 0: units supplied from S1 to C1
- x₁₂ ≥ 0: units supplied from S1 to C2
- x₂₁ ≥ 0: units supplied from S2 to C1
- x₂₂ ≥ 0: units supplied from S2 to C2

Mathematical Model:

Objective:
Minimize total cost (fixed + transportation):
\[
\min \quad 105.97\,y_1 + 85.31\,y_2 + 2358.39\,x_{11} + 1492.08\,x_{12} + 0.07\,x_{21} + 52.32\,x_{22}
\]

Subject to:

1. Demand satisfaction at each supermarket:
\[
x_{11} + x_{21} = 144 \quad \text{(C1 demand)}
\]
\[
x_{12} + x_{22} = 216 \quad \text{(C2 demand)}
\]

2. Supply only from activated suppliers (using a large constant M, e.g., M = 360, which is the sum of all demands):
\[
x_{11} + x_{12} \leq 360\,y_1
\]
\[
x_{21} + x_{22} \leq 360\,y_2
\]

3. Variable domains:
\[
y_1, y_2 \in \{0,1\}
\]
\[
x_{ij} \geq 0 \quad \forall i \in \{1,2\},\, j \in \{1,2\}
\]

Where:
- \( y_i \) indicates if supplier S_i is activated.
- \( x_{ij} \) is the amount supplied from supplier S_i to supermarket C_j.

All parameters, vectors, and matrices are explicitly given above. This model minimizes the sum of fixed activation costs and transportation costs while ensuring all supermarket demands are met and only activated suppliers can supply goods.