##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Parameters

- $I = \{S1, S2\}$ (Suppliers)
- $J = \{C1, C2\}$ (Supermarkets)
- Demands (current period):  
  $d_{C1} = 144$  
  $d_{C2} = 216$
- Fixed costs (current period):  
  $f_{S1} = 105.97$  
  $f_{S2} = 85.31$
- Transportation costs (current period):  
  $c_{S1,C1} = 2358.39$  
  $c_{S1,C2} = 1492.08$  
  $c_{S2,C1} = 0.07$  
  $c_{S2,C2} = 52.32$

##### Objective Function

\[
\min \left( 2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2} + 105.97\,y_{S1} + 85.31\,y_{S2} \right)
\]

##### Constraints

1. **Demand satisfaction:**  
   For each supermarket $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   That is,
   \[
   x_{S1,C1} + x_{S2,C1} = 144
   \]
   \[
   x_{S1,C2} + x_{S2,C2} = 216
   \]

2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   x_{ij} \leq d_j\,y_i \quad \forall j \in J
   \]
   That is, for all $i \in I$, $j \in J$:
   \[
   x_{ij} \leq d_j\,y_i
   \]
   (Inactive suppliers cannot ship any goods.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Full Model (with all parameters):

\[
\begin{align*}
\min\ & 2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2} + 105.97\,y_{S1} + 85.31\,y_{S2} \\
\text{s.t.}\quad
& x_{S1,C1} + x_{S2,C1} = 144 \\
& x_{S1,C2} + x_{S2,C2} = 216 \\
& x_{S1,C1} \leq 144\,y_{S1} \\
& x_{S1,C2} \leq 216\,y_{S1} \\
& x_{S2,C1} \leq 144\,y_{S2} \\
& x_{S2,C2} \leq 216\,y_{S2} \\
& x_{ij} \geq 0 \quad \forall i \in \{S1, S2\},\, j \in \{C1, C2\} \\
& y_{i} \in \{0,1\} \quad \forall i \in \{S1, S2\}
\end{align*}
\]

##### Retrieved Parameters

- Suppliers: S1, S2
- Supermarkets: C1, C2
- Demands: C1 = 144, C2 = 216
- Fixed costs: S1 = 105.97, S2 = 85.31
- Transportation costs:  
  S1→C1 = 2358.39, S1→C2 = 1492.08  
  S2→C1 = 0.07, S2→C2 = 52.32

All data is from the current period and preserves original identifiers and values.