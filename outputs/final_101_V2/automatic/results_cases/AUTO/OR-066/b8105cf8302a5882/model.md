##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Parameters

- $I = \{S1, S2\}$: Set of suppliers.
- $J = \{C1, C2\}$: Set of supermarkets.
- Demands: $d_{C1} = 144$, $d_{C2} = 216$.
- Fixed costs: $f_{S1} = 105.97$, $f_{S2} = 85.31$.
- Transportation costs:
  - $c_{S1,C1} = 2358.39$, $c_{S1,C2} = 1492.08$
  - $c_{S2,C1} = 0.07$, $c_{S2,C2} = 52.32$
- $M = d_{C1} + d_{C2} = 360$ (sufficiently large upper bound for each supplier's possible shipment).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

That is,

\[
\min \left[
2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2}
+ 105.97\,y_{S1} + 85.31\,y_{S2}
\right]
\]

##### Constraints

1. **Demand satisfaction:** Each supermarket must receive exactly its demand.
   - $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
   - $\begin{cases}
      x_{S1,C1} + x_{S2,C1} = 144 \\
      x_{S1,C2} + x_{S2,C2} = 216
     \end{cases}$

2. **Supplier activation:** No shipments from inactive suppliers.
   - $\sum_{j \in J} x_{ij} \leq M y_i,\quad \forall i \in I$
   - $\begin{cases}
      x_{S1,C1} + x_{S1,C2} \leq 360\,y_{S1} \\
      x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2}
     \end{cases}$

3. **Variable domains:**
   - $x_{ij} \geq 0$ (continuous), $\forall i \in I, j \in J$
   - $y_i \in \{0,1\}$, $\forall i \in I$

---

###### Retrieved Information

- Facilities: $I = \{S1, S2\}$
- Supermarkets: $J = \{C1, C2\}$
- Demands: $d_{C1} = 144$, $d_{C2} = 216$
- Fixed costs: $f_{S1} = 105.97$, $f_{S2} = 85.31$
- Transportation costs:
  - $c_{S1,C1} = 2358.39$, $c_{S1,C2} = 1492.08$
  - $c_{S2,C1} = 0.07$, $c_{S2,C2} = 52.32$
- $M = 360$

---

This model determines which suppliers to activate and how much each should ship to each supermarket, minimizing the total cost while meeting all supermarket demands.