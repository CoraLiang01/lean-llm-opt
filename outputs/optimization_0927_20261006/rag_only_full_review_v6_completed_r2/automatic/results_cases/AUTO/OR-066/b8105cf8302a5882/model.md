Let:
- F = {S1, S2} be the set of suppliers.
- C = {C1, C2} be the set of supermarkets.

Parameters:
- Fixed costs for activating supplier i ∈ F:
  - f_S1 = 105.97
  - f_S2 = 85.31
- Per-unit transportation costs from supplier i ∈ F to supermarket j ∈ C:
  - c_{S1,C1} = 2358.39
  - c_{S1,C2} = 1492.08
  - c_{S2,C1} = 0.07
  - c_{S2,C2} = 52.32
- Demand at each supermarket j ∈ C:
  - d_{C1} = 144
  - d_{C2} = 216

Decision Variables:
- y_i ∈ {0,1} for each i ∈ F: 1 if supplier i is activated, 0 otherwise.
- x_{ij} ≥ 0 for each i ∈ F, j ∈ C: quantity supplied from supplier i to supermarket j.

Mathematical Model:

Minimize total cost:
\[
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} c_{ij} x_{ij}
\]
That is,
\[
\min \left(105.97\,y_{S1} + 85.31\,y_{S2}\right) + \left(2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2}\right)
\]

Subject to:
1. Demand satisfaction at each supermarket:
   - \( x_{S1,C1} + x_{S2,C1} = 144 \)
   - \( x_{S1,C2} + x_{S2,C2} = 216 \)

2. Supply only from activated suppliers:
   - \( x_{S1,C1} + x_{S1,C2} \leq (144 + 216) y_{S1} = 360\,y_{S1} \)
   - \( x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2} \)

3. Variable domains:
   - \( y_{S1}, y_{S2} \in \{0,1\} \)
   - \( x_{ij} \geq 0 \) for all i ∈ F, j ∈ C

Where:
- \( y_{S1} \): 1 if supplier S1 is activated, 0 otherwise.
- \( y_{S2} \): 1 if supplier S2 is activated, 0 otherwise.
- \( x_{S1,C1} \): quantity supplied from S1 to C1, etc.

All parameters (fixed costs, transportation costs, demands) are as retrieved from the CSV files. This model determines which suppliers to activate and how to allocate supply to minimize the total cost of fixed activation and transportation, while meeting all supermarket demands.