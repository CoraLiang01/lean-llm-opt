##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier (facility) $i \in I$ to supermarket (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier (facility) $i$ is activated.

##### Parameters

- $I = \{S1, S2\}$ (set of suppliers/facilities)
- $J = \{C1, C2\}$ (set of supermarkets/customers)
- Demands: $d_{C1} = 144$, $d_{C2} = 216$
- Fixed costs: $f_{S1} = 105.97$, $f_{S2} = 85.31$
- Transportation costs:
  - $c_{S1,C1} = 2358.39$, $c_{S1,C2} = 1492.08$
  - $c_{S2,C1} = 0.07$, $c_{S2,C2} = 52.32$
- $M = d_{C1} + d_{C2} = 360$ (sufficiently large upper bound for linking constraints)

##### Objective Function

\[
\min \left[
    2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2}
  + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2}
  + 105.97\,y_{S1} + 85.31\,y_{S2}
\right]
\]

##### Constraints

1. **Demand satisfaction (each supermarket must receive its demand):**
   - $x_{S1,C1} + x_{S2,C1} = 144$
   - $x_{S1,C2} + x_{S2,C2} = 216$

2. **Supplier activation (no shipments from inactive suppliers):**
   - $x_{S1,C1} + x_{S1,C2} \leq 360\,y_{S1}$
   - $x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2}$

3. **Variable domains:**
   - $x_{ij} \geq 0$ for all $i \in I$, $j \in J$
   - $y_i \in \{0,1\}$ for all $i \in I$

##### All Parameters (as retrieved)

- Facilities: $I = \{S1, S2\}$
- Supermarkets: $J = \{C1, C2\}$
- Demands: $d_{C1} = 144$, $d_{C2} = 216$
- Fixed costs: $f_{S1} = 105.97$, $f_{S2} = 85.31$
- Transportation costs:
  - $c_{S1,C1} = 2358.39$, $c_{S1,C2} = 1492.08$
  - $c_{S2,C1} = 0.07$, $c_{S2,C2} = 52.32$
- $M = 360$

##### Model Summary

\[
\begin{align*}
\min\ & 2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2} + 105.97\,y_{S1} + 85.31\,y_{S2} \\
\text{s.t.}\quad
& x_{S1,C1} + x_{S2,C1} = 144 \\
& x_{S1,C2} + x_{S2,C2} = 216 \\
& x_{S1,C1} + x_{S1,C2} \leq 360\,y_{S1} \\
& x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2} \\
& x_{ij} \geq 0,\quad \forall i \in \{S1, S2\},\ j \in \{C1, C2\} \\
& y_i \in \{0,1\},\quad \forall i \in \{S1, S2\}
\end{align*}
\]

All parameters, vectors, and matrices are included as retrieved.