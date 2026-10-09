##### Decision Variables

$x_{ij} \geq 0$: quantity supplied from facility $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: 1 if facility $i$ is activated, 0 otherwise.

##### Parameters

- Facilities $I = \{\text{S1}, \text{S2}\}$
- Supermarkets $J = \{\text{C1}, \text{C2}\}$

- Demands:
  - $d_{\text{C1}} = 144$
  - $d_{\text{C2}} = 216$

- Fixed costs:
  - $f_{\text{S1}} = 105.97$
  - $f_{\text{S2}} = 85.31$

- Transportation costs:
  - $c_{\text{S1},\text{C1}} = 2358.39$
  - $c_{\text{S1},\text{C2}} = 1492.08$
  - $c_{\text{S2},\text{C1}} = 0.07$
  - $c_{\text{S2},\text{C2}} = 52.32$

##### Objective Function

\[
\min \left(
    2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}}
  + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}}
  + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}}
\right)
\]

##### Constraints

1. **Demand satisfaction:**  
   For each supermarket $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   That is,
   - $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144$
   - $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216$

2. **Facility activation:**  
   For each facility $i \in I$ and supermarket $j \in J$,
   \[
   x_{ij} \leq d_j\, y_i
   \]
   That is,
   - $x_{\text{S1},\text{C1}} \leq 144\, y_{\text{S1}}$
   - $x_{\text{S1},\text{C2}} \leq 216\, y_{\text{S1}}$
   - $x_{\text{S2},\text{C1}} \leq 144\, y_{\text{S2}}$
   - $x_{\text{S2},\text{C2}} \leq 216\, y_{\text{S2}}$

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All Parameters (from CSVs)

- Facilities: S1, S2
- Supermarkets: C1, C2
- Demands: $d_{\text{C1}} = 144$, $d_{\text{C2}} = 216$
- Fixed costs: $f_{\text{S1}} = 105.97$, $f_{\text{S2}} = 85.31$
- Transportation costs:
  - $c_{\text{S1},\text{C1}} = 2358.39$
  - $c_{\text{S1},\text{C2}} = 1492.08$
  - $c_{\text{S2},\text{C1}} = 0.07$
  - $c_{\text{S2},\text{C2}} = 52.32$

##### Model Summary

\[
\begin{align*}
\min\ & 2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}} + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}} \\
& + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}} \\
\text{s.t.}\quad
& x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144 \\
& x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216 \\
& x_{\text{S1},\text{C1}} \leq 144\, y_{\text{S1}} \\
& x_{\text{S1},\text{C2}} \leq 216\, y_{\text{S1}} \\
& x_{\text{S2},\text{C1}} \leq 144\, y_{\text{S2}} \\
& x_{\text{S2},\text{C2}} \leq 216\, y_{\text{S2}} \\
& x_{ij} \geq 0 \quad \forall i,j \\
& y_i \in \{0,1\} \quad \forall i \\
\end{align*}
\]