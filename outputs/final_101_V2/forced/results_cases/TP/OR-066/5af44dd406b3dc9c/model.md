##### Decision Variables

$y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise, for $i \in \{S1, S2\}$

$x_{ij} \geq 0$: quantity supplied from supplier $i$ to supermarket $j$, for $i \in \{S1, S2\}$, $j \in \{C1, C2\}$

##### Parameters

- Fixed costs:
  - $f_{S1} = 105.97$
  - $f_{S2} = 85.31$
- Transportation costs:
  - $c_{S1,C1} = 2358.39$
  - $c_{S1,C2} = 1492.08$
  - $c_{S2,C1} = 0.07$
  - $c_{S2,C2} = 52.32$
- Demands:
  - $d_{C1} = 144$
  - $d_{C2} = 216$

##### Objective Function

\[
\min\; 105.97\,y_{S1} + 85.31\,y_{S2} + 2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2}
\]

##### Constraints

1. Demand satisfaction for each supermarket:
   - $x_{S1,C1} + x_{S2,C1} = 144$
   - $x_{S1,C2} + x_{S2,C2} = 216$

2. Linking constraints (no supply from inactive suppliers):
   - $x_{S1,C1} + x_{S1,C2} \leq (144 + 216)\,y_{S1} = 360\,y_{S1}$
   - $x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2}$

3. Variable domains:
   - $y_{S1}, y_{S2} \in \{0,1\}$
   - $x_{ij} \geq 0$ for all $i \in \{S1, S2\}$, $j \in \{C1, C2\}$

##### Complete Model

\[
\begin{align*}
\min\quad & 105.97\,y_{S1} + 85.31\,y_{S2} + 2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2} \\
\text{s.t.}\quad
& x_{S1,C1} + x_{S2,C1} = 144 \\
& x_{S1,C2} + x_{S2,C2} = 216 \\
& x_{S1,C1} + x_{S1,C2} \leq 360\,y_{S1} \\
& x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2} \\
& y_{S1}, y_{S2} \in \{0,1\} \\
& x_{ij} \geq 0 \quad \forall i \in \{S1, S2\},\; j \in \{C1, C2\}
\end{align*}
\]