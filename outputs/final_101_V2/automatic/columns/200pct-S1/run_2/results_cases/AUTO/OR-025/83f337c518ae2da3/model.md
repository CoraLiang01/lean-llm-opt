##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Parameters

- Suppliers (Facilities): $I = \{\text{S1}, \text{S2}\}$
- Supermarkets (Customers): $J = \{\text{C1}, \text{C2}\}$
- Fixed costs:
  - $f_{\text{S1}} = 105.97$
  - $f_{\text{S2}} = 85.31$
- Demands:
  - $d_{\text{C1}} = 144$
  - $d_{\text{C2}} = 216$
- Transportation costs:
  - $c_{\text{S1},\text{C1}} = 2358.39$
  - $c_{\text{S1},\text{C2}} = 1492.08$
  - $c_{\text{S2},\text{C1}} = 0.07$
  - $c_{\text{S2},\text{C2}} = 52.32$
- $M = \sum_{j \in J} d_j = 144 + 216 = 360$

##### Objective Function

\[
\min \left(
    2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}}
  + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}}
  + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}}
\right)
\]

##### Constraints

1. Supermarket demand satisfaction:
   - $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144$
   - $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216$

2. Supplier activation (inactive suppliers cannot ship):
   - $x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}}$
   - $x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}}$

3. Variable domains:
   - $x_{ij} \geq 0$ for all $i \in I$, $j \in J$
   - $y_i \in \{0,1\}$ for all $i \in I$

##### Complete Model

\[
\begin{align*}
\min\quad & 2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}}
  + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}}
  + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}} \\
\text{s.t.}\quad
& x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144 \\
& x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216 \\
& x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}} \\
& x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}} \\
& x_{ij} \geq 0 \quad \forall i \in \{\text{S1},\text{S2}\},\ j \in \{\text{C1},\text{C2}\} \\
& y_i \in \{0,1\} \quad \forall i \in \{\text{S1},\text{S2}\}
\end{align*}
\]

###### Retrieved Information

- Suppliers: S1, S2
- Supermarkets: C1, C2
- Fixed costs: S1: 105.97, S2: 85.31
- Demands: C1: 144, C2: 216
- Transportation costs:
    - S1→C1: 2358.39, S1→C2: 1492.08
    - S2→C1: 0.07, S2→C2: 52.32
- $M = 360$ (total demand, used as a big-M upper bound for supplier activation constraints)