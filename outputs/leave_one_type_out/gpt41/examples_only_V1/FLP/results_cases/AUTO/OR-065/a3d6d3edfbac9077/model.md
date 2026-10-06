Mathematical Model for Bandcamp Inventory Replenishment via Warehouses

Sets:
- Let F = {S1, S2, S3} be the set of warehouses.
- Let C = {C1, C2, C3} be the set of musicians/bands (customers).

Parameters:
- Fixed costs for opening each warehouse:
    - f₁ = 102.33 (S1)
    - f₂ = 94.92  (S2)
    - f₃ = 91.83  (S3)
- Demand for each customer:
    - d₁ = 1083   (C1)
    - d₂ = 776    (C2)
    - d₃ = 16214  (C3)
- Transportation costs per unit from warehouse i to customer j (c_{ij}):
    - c_{11} = 1506.22 (S1 → C1)
    - c_{12} = 70.90   (S1 → C2)
    - c_{13} = 8.44    (S1 → C3)
    - c_{21} = 1732.65 (S2 → C1)
    - c_{22} = 1780.72 (S2 → C2)
    - c_{23} = 567.44  (S2 → C3)
    - c_{31} = 115.66  (S3 → C1)
    - c_{32} = 100.76  (S3 → C2)
    - c_{33} = 64.68   (S3 → C3)

Decision Variables:
- y_i ∈ {0,1} for i ∈ F: 1 if warehouse i is opened, 0 otherwise.
- x_{ij} ≥ 0 for i ∈ F, j ∈ C: quantity of goods shipped from warehouse i to customer j.

Mathematical Formulation:

Objective:
Minimize total cost (fixed + transportation):
\[
\min \left[
    102.33\,y_1 + 94.92\,y_2 + 91.83\,y_3
    + 1506.22\,x_{11} + 70.90\,x_{12} + 8.44\,x_{13}
    + 1732.65\,x_{21} + 1780.72\,x_{22} + 567.44\,x_{23}
    + 115.66\,x_{31} + 100.76\,x_{32} + 64.68\,x_{33}
\right]
\]

Subject to:

1. Demand satisfaction for each customer:
\[
x_{1j} + x_{2j} + x_{3j} = d_j \quad \forall j \in \{1,2,3\}
\]
Explicitly:
\[
x_{11} + x_{21} + x_{31} = 1083
\]
\[
x_{12} + x_{22} + x_{32} = 776
\]
\[
x_{13} + x_{23} + x_{33} = 16214
\]

2. Supply only from open warehouses:
\[
x_{ij} \leq d_j\,y_i \quad \forall i \in \{1,2,3\},\; j \in \{1,2,3\}
\]

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in \{1,2,3\}
\]
\[
x_{ij} \geq 0 \quad \forall i \in \{1,2,3\},\; j \in \{1,2,3\}
\]

Where:
- i = 1,2,3 correspond to S1, S2, S3
- j = 1,2,3 correspond to C1, C2, C3

All parameters (fixed costs, transportation costs, demands) are explicitly stated above. This model determines which warehouses to open and how much each should supply to each musician/band to minimize total cost while meeting all demands.