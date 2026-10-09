##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier (facility) $i \in I$ to supermarket (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Parameters

- $I = \{S1, S2\}$ (Suppliers)
- $J = \{C1, C2\}$ (Supermarkets)
- Demand:
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
- $M = d_{C1} + d_{C2} = 360$ (sufficiently large upper bound for linking constraints)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

That is,

\[
\min \left[
2358.39\, x_{S1,C1} + 1492.08\, x_{S1,C2} + 0.07\, x_{S2,C1} + 52.32\, x_{S2,C2}
+ 105.97\, y_{S1} + 85.31\, y_{S2}
\right]
\]

##### Constraints

1. **Demand satisfaction (each supermarket receives exactly its demand):**
   \[
   x_{S1,C1} + x_{S2,C1} = 144
   \]
   \[
   x_{S1,C2} + x_{S2,C2} = 216
   \]

2. **Supplier activation (no shipments from inactive suppliers):**
   \[
   x_{S1,C1} + x_{S1,C2} \leq 360\, y_{S1}
   \]
   \[
   x_{S2,C1} + x_{S2,C2} \leq 360\, y_{S2}
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All Parameters (Current Period)

- Suppliers: $I = \{S1, S2\}$
- Supermarkets: $J = \{C1, C2\}$
- Demand: $d_{C1} = 144$, $d_{C2} = 216$
- Fixed costs: $f_{S1} = 105.97$, $f_{S2} = 85.31$
- Transportation costs:
  - $c_{S1,C1} = 2358.39$
  - $c_{S1,C2} = 1492.08$
  - $c_{S2,C1} = 0.07$
  - $c_{S2,C2} = 52.32$
- $M = 360$

##### Complete Mathematical Model

\[
\begin{align*}
\min\quad & 2358.39\, x_{S1,C1} + 1492.08\, x_{S1,C2} + 0.07\, x_{S2,C1} + 52.32\, x_{S2,C2} \\
&\quad + 105.97\, y_{S1} + 85.31\, y_{S2} \\
\text{s.t.}\quad
& x_{S1,C1} + x_{S2,C1} = 144 \\
& x_{S1,C2} + x_{S2,C2} = 216 \\
& x_{S1,C1} + x_{S1,C2} \leq 360\, y_{S1} \\
& x_{S2,C1} + x_{S2,C2} \leq 360\, y_{S2} \\
& x_{ij} \geq 0 \quad \forall i \in \{S1, S2\},\, j \in \{C1, C2\} \\
& y_{i} \in \{0,1\} \quad \forall i \in \{S1, S2\}
\end{align*}
\]