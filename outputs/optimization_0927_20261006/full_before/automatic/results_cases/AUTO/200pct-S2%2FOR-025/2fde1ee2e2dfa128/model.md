##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Parameters

- $I = \{S1, S2\}$ (Suppliers)
- $J = \{C1, C2\}$ (Supermarkets)
- Demands: $d_{C1} = 144$, $d_{C2} = 216$
- Fixed costs: $f_{S1} = 105.97$, $f_{S2} = 85.31$
- Transportation costs:
  - $c_{S1,C1} = 2358.39$, $c_{S1,C2} = 1492.08$
  - $c_{S2,C1} = 0.07$, $c_{S2,C2} = 52.32$
- $M = d_{C1} + d_{C2} = 360$ (sufficiently large upper bound for linking constraints)

##### Objective Function

\[
\min \left(
    2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2}
  + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2}
  + 105.97\,y_{S1} + 85.31\,y_{S2}
\right)
\]

##### Constraints

1. **Demand satisfaction (each supermarket receives its demand):**
   \[
   x_{S1,C1} + x_{S2,C1} = 144
   \]
   \[
   x_{S1,C2} + x_{S2,C2} = 216
   \]

2. **Supplier activation (no shipments from inactive suppliers):**
   \[
   x_{S1,C1} + x_{S1,C2} \leq 360\,y_{S1}
   \]
   \[
   x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2}
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \forall i \in \{S1, S2\},\ j \in \{C1, C2\}
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in \{S1, S2\}
   \]

##### All Parameters (as retrieved)

- Suppliers: S1, S2
- Supermarkets: C1, C2
- Demands: C1 = 144, C2 = 216
- Fixed costs: S1 = 105.97, S2 = 85.31
- Transportation costs:
    - S1 to C1: 2358.39
    - S1 to C2: 1492.08
    - S2 to C1: 0.07
    - S2 to C2: 52.32
- $M = 360$

This model determines which suppliers to activate and how much each should ship to each supermarket, minimizing the total fixed and transportation costs while meeting all supermarket demands.