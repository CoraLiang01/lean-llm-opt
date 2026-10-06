Let us define the following sets, parameters, and variables based on the provided CSV data:

Sets:
- Suppliers (Facilities): \( I = \{S1, S2\} \)
- Supermarkets (Customers): \( J = \{C1, C2\} \)

Parameters:
- Fixed costs for activating each supplier:
  - \( f_{S1} = 105.97 \)
  - \( f_{S2} = 85.31 \)
- Per-unit transportation costs from each supplier to each supermarket:
  - \( c_{S1,C1} = 2358.39 \)
  - \( c_{S1,C2} = 1492.08 \)
  - \( c_{S2,C1} = 0.07 \)
  - \( c_{S2,C2} = 52.32 \)
- Demand at each supermarket:
  - \( d_{C1} = 144 \)
  - \( d_{C2} = 216 \)

Decision Variables:
- \( y_i \in \{0,1\} \) for \( i \in I \): 1 if supplier \( i \) is activated, 0 otherwise.
- \( x_{ij} \geq 0 \) for \( i \in I, j \in J \): Amount supplied from supplier \( i \) to supermarket \( j \).

Mathematical Model:

Objective:
\[
\min \left( \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \right)
\]
That is,
\[
\min \left( 105.97\,y_{S1} + 85.31\,y_{S2} + 2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2} \right)
\]

Subject to:

1. Demand satisfaction at each supermarket:
   \[
   x_{S1,C1} + x_{S2,C1} = 144
   \]
   \[
   x_{S1,C2} + x_{S2,C2} = 216
   \]

2. Supply only from activated suppliers:
   \[
   x_{S1,C1} + x_{S1,C2} \leq (144 + 216) y_{S1} = 360\,y_{S1}
   \]
   \[
   x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2}
   \]
   (Here, 360 is the total demand; this ensures that if \( y_i = 0 \), then \( x_{ij} = 0 \) for all \( j \).)

3. Variable domains:
   \[
   y_{S1}, y_{S2} \in \{0,1\}
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   \]

Summary of all parameters:

- Suppliers: \( S1, S2 \)
- Supermarkets: \( C1, C2 \)
- Fixed costs: \( f_{S1} = 105.97, f_{S2} = 85.31 \)
- Transportation cost matrix:
  \[
  \begin{array}{c|cc}
    & C1 & C2 \\
    \hline
    S1 & 2358.39 & 1492.08 \\
    S2 & 0.07 & 52.32 \\
  \end{array}
  \]
- Demands: \( d_{C1} = 144, d_{C2} = 216 \)

This model determines which suppliers to activate (\( y_i \)) and how much each supplier should deliver to each supermarket (\( x_{ij} \)) to minimize the total cost, while satisfying all supermarket demands.