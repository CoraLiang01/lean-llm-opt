Let:
- Suppliers: F = {S1, S2}
- Supermarkets (customers): C = {C1, C2}
- Fixed costs: f = [f1, f2] = [105.97, 85.31]
- Demands: d = [d1, d2] = [144, 216]
- Transportation costs: t = [[2358.39, 1492.08], [0.07, 52.32]]
  - t_{ij} is the cost from supplier S_i to customer C_j

Decision variables:
- y_i ∈ {0,1} for i ∈ {1,2}: y_i = 1 if supplier S_i is activated, 0 otherwise
- x_{ij} ≥ 0: amount supplied from supplier S_i to customer C_j

Mathematical Model:

Minimize total cost:
\[
\min \left( 105.97\,y_1 + 85.31\,y_2 + 2358.39\,x_{11} + 1492.08\,x_{12} + 0.07\,x_{21} + 52.32\,x_{22} \right)
\]

Subject to:
1. Demand satisfaction for each customer:
   \[
   x_{1j} + x_{2j} = d_j \quad \forall j \in \{1,2\}
   \]
   That is,
   \[
   x_{11} + x_{21} = 144
   \]
   \[
   x_{12} + x_{22} = 216
   \]

2. Supply only from activated suppliers:
   \[
   x_{ij} \leq d_j\,y_i \quad \forall i \in \{1,2\},\, j \in \{1,2\}
   \]
   That is,
   \[
   x_{11} \leq 144\,y_1
   \]
   \[
   x_{12} \leq 216\,y_1
   \]
   \[
   x_{21} \leq 144\,y_2
   \]
   \[
   x_{22} \leq 216\,y_2
   \]

3. Variable domains:
   \[
   y_i \in \{0,1\} \quad \forall i \in \{1,2\}
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in \{1,2\},\, j \in \{1,2\}
   \]

Parameter summary:
- Suppliers: S1, S2
- Customers: C1, C2
- Fixed costs: f = [105.97, 85.31]
- Demands: d = [144, 216]
- Transportation cost matrix:
  \[
  t = \begin{bmatrix}
  2358.39 & 1492.08 \\
  0.07 & 52.32
  \end{bmatrix}
  \]

Variables:
- y_1, y_2 ∈ {0,1}
- x_{11}, x_{12}, x_{21}, x_{22} ≥ 0

Objective:
\[
\min \left( 105.97\,y_1 + 85.31\,y_2 + 2358.39\,x_{11} + 1492.08\,x_{12} + 0.07\,x_{21} + 52.32\,x_{22} \right)
\]

Subject to:
\[
x_{11} + x_{21} = 144
\]
\[
x_{12} + x_{22} = 216
\]
\[
x_{11} \leq 144\,y_1
\]
\[
x_{12} \leq 216\,y_1
\]
\[
x_{21} \leq 144\,y_2
\]
\[
x_{22} \leq 216\,y_2
\]
\[
y_1, y_2 \in \{0,1\}
\]
\[
x_{ij} \geq 0
\]

This model determines which suppliers to activate and how much each should supply to each supermarket to minimize the total cost.