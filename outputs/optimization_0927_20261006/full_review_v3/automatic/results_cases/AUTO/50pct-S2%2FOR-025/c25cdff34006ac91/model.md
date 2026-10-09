##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. Supermarket demand: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j\in J$
2. Supplier activation: $\sum_{j\in J} x_{ij} \leq M y_i,\quad \forall i\in I$
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

Where $M = \sum_{j\in J} d_j = 144 + 216 = 360$.

##### Parameters

- Suppliers $I = \{S1, S2\}$
- Supermarkets $J = \{C1, C2\}$

- Demands:
  - $d_{C1} = 144$
  - $d_{C2} = 216$

- Fixed costs:
  - $f_{S1} = 105.97$
  - $f_{S2} = 85.31$

- Transportation costs (per unit):

|         | C1      | C2      |
|---------|---------|---------|
| S1      | 2358.39 | 1492.08 |
| S2      | 0.07    | 52.32   |

##### Full Model

\[
\begin{align*}
\min\ & 2358.39\,x_{S1,C1} + 1492.08\,x_{S1,C2} + 0.07\,x_{S2,C1} + 52.32\,x_{S2,C2} + 105.97\,y_{S1} + 85.31\,y_{S2} \\
\text{s.t.}\quad
& x_{S1,C1} + x_{S2,C1} = 144 \\
& x_{S1,C2} + x_{S2,C2} = 216 \\
& x_{S1,C1} + x_{S1,C2} \leq 360\,y_{S1} \\
& x_{S2,C1} + x_{S2,C2} \leq 360\,y_{S2} \\
& x_{S1,C1},\ x_{S1,C2},\ x_{S2,C1},\ x_{S2,C2} \geq 0 \\
& y_{S1},\ y_{S2} \in \{0,1\}
\end{align*}
\]

###### Retrieved Information

{
  "suppliers": [
    {"id": "S1", "facility_staff_count": 65, "fixed_costs": 105.97, "operations_region": "West", "annual_inspection_count": 1},
    {"id": "S2", "facility_staff_count": 20, "fixed_costs": 85.31, "operations_region": "North", "annual_inspection_count": 4}
  ],
  "supermarkets": [
    {"id": "C1", "customer_support_ticket_count": 3, "demand": 144},
    {"id": "C2", "customer_support_ticket_count": 1, "demand": 216}
  ],
  "fixed_cost": {
    "S1": 105.97,
    "S2": 85.31
  },
  "demand": {
    "C1": 144,
    "C2": 216
  },
  "cost": {
    "S1": {"C1": 2358.39, "C2": 1492.08},
    "S2": {"C1": 0.07, "C2": 52.32}
  },
  "M": 360
}