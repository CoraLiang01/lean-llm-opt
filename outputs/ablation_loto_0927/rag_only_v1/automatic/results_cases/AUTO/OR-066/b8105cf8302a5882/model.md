Let us define the following sets, parameters, and variables based on the provided CSV data:

Sets:
- Facilities (Suppliers): \( I = \{S1, S2\} \)
- Customers (Supermarkets): \( J = \{C1, C2\} \)

Parameters:
- Fixed costs for opening each supplier:
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

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
= \quad & 105.97\,y_{S1} + 85.31\,y_{S2} \\
& + 2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} \\
& + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2}
\end{align*}
\]

Subject to:

1. Demand satisfaction at each supermarket:
   \[
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   \]
   Specifically:
   \[
   x_{S1,C1} + x_{S2,C1} = 144
   \]
   \[
   x_{S1,C2} + x_{S2,C2} = 216
   \]

2. Supply only from open suppliers:
   \[
   x_{ij} \leq d_j y_i \quad \forall i \in I, j \in J
   \]
   Specifically:
   \[
   x_{S1,C1} \leq 144\,y_{S1}
   \]
   \[
   x_{S1,C2} \leq 216\,y_{S1}
   \]
   \[
   x_{S2,C1} \leq 144\,y_{S2}
   \]
   \[
   x_{S2,C2} \leq 216\,y_{S2}
   \]

3. Binary and non-negativity constraints:
   \[
   y_{i} \in \{0,1\} \quad \forall i \in I
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   \]

Summary of parameters:

- Facilities: \( I = \{S1, S2\} \)
- Customers: \( J = \{C1, C2\} \)
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
- Demand vector: \( d_{C1} = 144, d_{C2} = 216 \)

This model determines which suppliers to activate (minimizing fixed and transportation costs) and how much each supplier should deliver to each supermarket to satisfy all demands.