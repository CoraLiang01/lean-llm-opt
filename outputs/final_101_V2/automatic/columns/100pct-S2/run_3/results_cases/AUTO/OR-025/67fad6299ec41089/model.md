##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Parameters

- $I = \{S1, S2\}$ (Supplier/Facility IDs)
- $J = \{C1, C2\}$ (Supermarket/Customer IDs)
- Demands:
  - $d_{C1} = 144$
  - $d_{C2} = 216$
- Fixed costs:
  - $f_{S1} = 105.97$
  - $f_{S2} = 85.31$
- Transportation costs:
  - $c_{S1,C1} = 2358.39$
  - $c_{S1,C2} = 1492.08$
  - $c_{S2,C1} = 0.07$
  - $c_{S2,C2} = 52.32$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]
That is,
\[
\min \left(2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2} + 105.97\,y_{S1} + 85.31\,y_{S2}\right)
\]

##### Constraints

1. **Demand satisfaction:**  
   For each supermarket $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   Explicitly:
   - $x_{S1,C1} + x_{S2,C1} = 144$
   - $x_{S1,C2} + x_{S2,C2} = 216$

2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M\,y_i
   \]
   where $M = \sum_{j \in J} d_j = 144 + 216 = 360$ (a valid upper bound since no explicit capacity is given).

   Explicitly:
   - $x_{S1,C1} + x_{S1,C2} \leq 360\,y_{S1}$
   - $x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2}$

3. **Variable domains:**
   - $x_{ij} \geq 0$ (continuous)
   - $y_i \in \{0,1\}$

##### Complete Model

\[
\begin{align*}
\min\quad & 2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2} + 105.97\,y_{S1} + 85.31\,y_{S2} \\
\text{s.t.}\quad
& x_{S1,C1} + x_{S2,C1} = 144 \\
& x_{S1,C2} + x_{S2,C2} = 216 \\
& x_{S1,C1} + x_{S1,C2} \leq 360\,y_{S1} \\
& x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2} \\
& x_{ij} \geq 0,\quad \forall i \in \{S1, S2\},\ j \in \{C1, C2\} \\
& y_i \in \{0,1\},\quad \forall i \in \{S1, S2\}
\end{align*}
\]

##### Retrieved Parameters

- Suppliers: S1, S2
- Supermarkets: C1, C2
- Demands: C1 = 144, C2 = 216
- Fixed costs: S1 = 105.97, S2 = 85.31
- Transportation costs: S1→C1 = 2358.39, S1→C2 = 1492.08, S2→C1 = 0.07, S2→C2 = 52.32
- $M = 360$