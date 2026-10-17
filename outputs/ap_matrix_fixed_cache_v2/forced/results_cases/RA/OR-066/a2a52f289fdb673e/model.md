Let:
- $y_i \in \{0,1\}$ indicate whether supplier $S_i$ is activated (incurs fixed cost).
- $x_{ij} \geq 0$ is the amount supplied from supplier $S_i$ to customer $C_j$.

Parameters:
- Suppliers: $S_1$, $S_2$
- Customers: $C_1$, $C_2$
- Fixed costs: $f_{S_1} = 105.97$, $f_{S_2} = 85.31$
- Transportation costs:
  - $c_{S_1,C_1} = 2358.39$, $c_{S_1,C_2} = 1492.08$
  - $c_{S_2,C_1} = 0.07$, $c_{S_2,C_2} = 52.32$
- Demands: $d_{C_1} = 144$, $d_{C_2} = 216$

Model:

Minimize total cost:
$$
\min \quad 105.97\,y_{S_1} + 85.31\,y_{S_2} + 2358.39\,x_{S_1,C_1} + 1492.08\,x_{S_1,C_2} + 0.07\,x_{S_2,C_1} + 52.32\,x_{S_2,C_2}
$$

Subject to:

1. Demand satisfaction for each customer:
\begin{align*}
x_{S_1,C_1} + x_{S_2,C_1} &= 144 \\
x_{S_1,C_2} + x_{S_2,C_2} &= 216 \\
\end{align*}

2. Supply only from activated suppliers:
\begin{align*}
x_{S_1,C_1} + x_{S_1,C_2} &\leq (144 + 216)\,y_{S_1} = 360\,y_{S_1} \\
x_{S_2,C_1} + x_{S_2,C_2} &\leq 360\,y_{S_2} \\
\end{align*}

3. Variable domains:
\begin{align*}
& y_{S_1}, y_{S_2} \in \{0,1\} \\
& x_{S_i,C_j} \geq 0 \quad \forall i \in \{S_1, S_2\},\ j \in \{C_1, C_2\}
\end{align*}

All data, identifiers, and coefficients are as retrieved and in original order.