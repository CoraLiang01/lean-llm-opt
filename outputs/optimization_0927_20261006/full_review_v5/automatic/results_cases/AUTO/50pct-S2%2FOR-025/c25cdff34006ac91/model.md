##### Parameters

- Suppliers: $I = \{S1, S2\}$
- Supermarkets: $J = \{C1, C2\}$

- Demand:
  - $d_{C1} = 144$
  - $d_{C2} = 216$

- Fixed costs:
  - $f_{S1} = 105.97$
  - $f_{S2} = 85.31$

- Transportation costs ($c_{ij}$):

|         | C1      | C2      |
|---------|---------|---------|
| S1      | 2358.39 | 1492.08 |
| S2      | 0.07    | 52.32   |

- Total demand: $M = d_{C1} + d_{C2} = 360$

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous)
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Supermarket demand satisfaction**:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   - For $C1$: $x_{S1,C1} + x_{S2,C1} = 144$
   - For $C2$: $x_{S1,C2} + x_{S2,C2} = 216$

2. **Supplier activation constraint**:
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   - For $S1$: $x_{S1,C1} + x_{S1,C2} \leq 360 y_{S1}$
   - For $S2$: $x_{S2,C1} + x_{S2,C2} \leq 360 y_{S2}$

3. **Variable domains**:
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Parameter Tables

- Demand:
  - $d_{C1} = 144$
  - $d_{C2} = 216$

- Fixed costs:
  - $f_{S1} = 105.97$
  - $f_{S2} = 85.31$

- Transportation costs:

|         | C1      | C2      |
|---------|---------|---------|
| S1      | 2358.39 | 1492.08 |
| S2      | 0.07    | 52.32   |

- $M = 360$

##### Sets

- $I = \{S1, S2\}$
- $J = \{C1, C2\}$

---

This model determines which suppliers to activate and how much each should ship to each supermarket, minimizing total fixed and transportation costs, while ensuring all supermarket demands are met and inactive suppliers do not ship goods.