##### Decision Variables

$x_{ij} \geq 0$: quantity supplied from supplier (facility) $i \in I$ to supermarket (customer) $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether supplier (facility) $i$ is activated.

##### Parameters

- $I = \{\text{S1}, \text{S2}\}$ (set of suppliers/facilities)
- $J = \{\text{C1}, \text{C2}\}$ (set of supermarkets/customers)

- Demand:
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
   Each supermarket must receive exactly its demand.
   - $x_{\text{S1},j} + x_{\text{S2},j} = d_j,\quad \forall j \in \{\text{C1}, \text{C2}\}$
     - $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144$
     - $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216$

2. **Facility activation:**  
   No supplier can ship unless activated. Let $M = d_{\text{C1}} + d_{\text{C2}} = 360$ (a valid upper bound).
   - $x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}}$
   - $x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}}$

3. **Variable domains:**
   - $x_{ij} \geq 0$ (continuous), for all $i \in I$, $j \in J$
   - $y_i \in \{0,1\}$, for all $i \in I$

##### Full Model

\[
\begin{align*}
\min\quad & 2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}} + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}} \\
&\quad + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}} \\
\text{s.t.}\quad
& x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144 \\
& x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216 \\
& x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}} \\
& x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}} \\
& x_{ij} \geq 0,\quad \forall i \in \{\text{S1},\text{S2}\},\ j \in \{\text{C1},\text{C2}\} \\
& y_i \in \{0,1\},\quad \forall i \in \{\text{S1},\text{S2}\}
\end{align*}
\]

##### Retrieved Parameters

- Facilities: S1, S2
- Customers: C1, C2
- Demand: C1 = 144, C2 = 216
- Fixed costs: S1 = 105.97, S2 = 85.31
- Transportation costs:
    - S1→C1: 2358.39, S1→C2: 1492.08
    - S2→C1: 0.07, S2→C2: 52.32
- $M = 360$