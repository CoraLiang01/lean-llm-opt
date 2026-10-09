Sets:
- Suppliers: \( I = \{S1, S2\} \)
- Supermarkets: \( J = \{C1, C2\} \)

Parameters:
- Fixed costs: \( f_{S1} = 105.97 \), \( f_{S2} = 85.31 \)
- Demands: \( d_{C1} = 144 \), \( d_{C2} = 216 \)
- Transportation costs:
  - \( c_{S1,C1} = 2358.39 \)
  - \( c_{S1,C2} = 1492.08 \)
  - \( c_{S2,C1} = 0.07 \)
  - \( c_{S2,C2} = 52.32 \)

Variables:
- \( y_{S1}, y_{S2} \in \{0,1\} \)
- \( x_{S1,C1}, x_{S1,C2}, x_{S2,C1}, x_{S2,C2} \geq 0 \)

Model:
\[
\begin{align*}
\text{Minimize} \quad & 105.97\,y_{S1} + 85.31\,y_{S2} + 2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2} \\
\text{subject to} \quad
& x_{S1,C1} + x_{S2,C1} = 144 \\
& x_{S1,C2} + x_{S2,C2} = 216 \\
& x_{S1,C1} \leq 144\,y_{S1} \\
& x_{S1,C2} \leq 216\,y_{S1} \\
& x_{S2,C1} \leq 144\,y_{S2} \\
& x_{S2,C2} \leq 216\,y_{S2} \\
& y_{S1}, y_{S2} \in \{0,1\} \\
& x_{S1,C1}, x_{S1,C2}, x_{S2,C1}, x_{S2,C2} \geq 0
\end{align*}
\]