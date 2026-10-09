##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Parameters

- Suppliers: $I = \{S1, S2\}$
- Supermarkets: $J = \{C1, C2\}$
- Demands:
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
- $M = d_{C1} + d_{C2} = 360$ (sufficiently large upper bound for each supplier's total shipment)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]
That is,
\[
\min \left[2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2} + 105.97\,y_{S1} + 85.31\,y_{S2}\right]
\]

##### Constraints

1. Supermarket demand satisfaction:
   \[
   x_{S1,C1} + x_{S2,C1} = 144
   \]
   \[
   x_{S1,C2} + x_{S2,C2} = 216
   \]

2. Supplier activation (no shipment from inactive suppliers):
   \[
   x_{S1,C1} + x_{S1,C2} \leq 360\,y_{S1}
   \]
   \[
   x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2}
   \]

3. Variable domains:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Retrieved Information

{
  "suppliers": ["S1", "S2"],
  "supermarkets": ["C1", "C2"],
  "demand": {
    "C1": 144,
    "C2": 216
  },
  "fixed_cost": {
    "S1": 105.97,
    "S2": 85.31
  },
  "cost": {
    "S1": {
      "C1": 2358.39,
      "C2": 1492.08
    },
    "S2": {
      "C1": 0.07,
      "C2": 52.32
    }
  },
  "M": 360
}