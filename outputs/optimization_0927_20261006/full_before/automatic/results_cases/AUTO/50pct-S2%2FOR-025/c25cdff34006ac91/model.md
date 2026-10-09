##### Decision Variables

$x_{ij} \geq 0$: Quantity shipped from supplier (facility) $i \in I$ to supermarket (customer) $j \in J$ (continuous).
$y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated, 0 otherwise.

##### Parameters

- $I = \{S1, S2\}$ (Facilities/Suppliers)
- $J = \{C1, C2\}$ (Supermarkets/Customers)

- Fixed costs:
  - $f_{S1} = 105.97$
  - $f_{S2} = 85.31$

- Transportation costs:
  - $c_{S1,C1} = 2358.39$
  - $c_{S1,C2} = 1492.08$
  - $c_{S2,C1} = 0.07$
  - $c_{S2,C2} = 52.32$

- Demands:
  - $d_{C1} = 144$
  - $d_{C2} = 216$

- Big-M for each facility: $M = \sum_{j \in J} d_j = 144 + 216 = 360$

##### Objective Function

\[
\min \left(
    2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2}
  + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2}
  + 105.97\,y_{S1} + 85.31\,y_{S2}
\right)
\]

##### Constraints

1. **Demand satisfaction (each supermarket must receive its demand):**
   \[
   x_{S1,C1} + x_{S2,C1} = 144
   \]
   \[
   x_{S1,C2} + x_{S2,C2} = 216
   \]

2. **Facility activation (no shipments from inactive suppliers):**
   \[
   x_{S1,C1} + x_{S1,C2} \leq 360\,y_{S1}
   \]
   \[
   x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2}
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All Parameters (as retrieved)

- Facilities: $I = \{S1, S2\}$
- Supermarkets: $J = \{C1, C2\}$
- Fixed costs: $f_{S1} = 105.97$, $f_{S2} = 85.31$
- Transportation costs:
  - $c_{S1,C1} = 2358.39$, $c_{S1,C2} = 1492.08$
  - $c_{S2,C1} = 0.07$, $c_{S2,C2} = 52.32$
- Demands: $d_{C1} = 144$, $d_{C2} = 216$
- $M = 360$ (sum of all demands)

##### Model Summary

Minimize total cost (fixed + transportation), subject to:
- Each supermarket’s demand is met,
- No shipments from inactive suppliers,
- All variables in their proper domains.