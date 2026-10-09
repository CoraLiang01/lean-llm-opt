##### Decision Variables

$x_{ij} \geq 0$: Quantity supplied from facility $i \in I$ to supermarket $j \in J$ (continuous, nonnegative).  
$y_i \in \{0,1\}$: 1 if facility $i$ is activated, 0 otherwise.

##### Parameters

- Facilities: $I = \{\text{S1}, \text{S2}\}$
- Supermarkets: $J = \{\text{C1}, \text{C2}\}$

- Demand (current period):  
  $d_{\text{C1}} = 144$  
  $d_{\text{C2}} = 216$

- Fixed costs (current period):  
  $f_{\text{S1}} = 105.97$  
  $f_{\text{S2}} = 85.31$

- Per-unit transportation costs (current period):  
  $c_{\text{S1},\text{C1}} = 2358.39$  
  $c_{\text{S1},\text{C2}} = 1492.08$  
  $c_{\text{S2},\text{C1}} = 0.07$  
  $c_{\text{S2},\text{C2}} = 52.32$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]
That is,
\[
\min \left(2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}} + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}} + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}}\right)
\]

##### Constraints

1. **Demand satisfaction:**  
   Each supermarket must receive exactly its demand.
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   Specifically:
   \[
   x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144
   \]
   \[
   x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216
   \]

2. **Facility activation:**  
   No shipments from inactive facilities.
   \[
   \sum_{j \in J} x_{ij} \leq M\,y_i, \quad \forall i \in I
   \]
   Where $M = d_{\text{C1}} + d_{\text{C2}} = 360$.

   That is:
   \[
   x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}}
   \]
   \[
   x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}}
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Complete Model Data

- Facilities: $I = \{\text{S1}, \text{S2}\}$
- Supermarkets: $J = \{\text{C1}, \text{C2}\}$
- Demand: $d_{\text{C1}} = 144$, $d_{\text{C2}} = 216$
- Fixed costs: $f_{\text{S1}} = 105.97$, $f_{\text{S2}} = 85.31$
- Transportation costs:
  - $c_{\text{S1},\text{C1}} = 2358.39$
  - $c_{\text{S1},\text{C2}} = 1492.08$
  - $c_{\text{S2},\text{C1}} = 0.07$
  - $c_{\text{S2},\text{C2}} = 52.32$
- $M = 360$

##### Model Summary

\[
\begin{align*}
\min\ & 2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}} + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}} + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}} \\
\text{s.t.}\quad
& x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144 \\
& x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216 \\
& x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}} \\
& x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}} \\
& x_{ij} \geq 0,\quad y_i \in \{0,1\}
\end{align*}
\]