##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods supplied from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise (binary).

##### Parameters

- Warehouses $I = \{S1, S2, S3\}$
- Musicians/Bands $J = \{C1, C2, C3\}$

- Demands:
  - $d_{C1} = 1083$
  - $d_{C2} = 776$
  - $d_{C3} = 16214$

- Fixed opening costs:
  - $f_{S1} = 102.33$
  - $f_{S2} = 94.92$
  - $f_{S3} = 91.83$

- Transportation costs per unit:
  - $c_{S1,C1} = 1506.22$, $c_{S1,C2} = 70.9$, $c_{S1,C3} = 8.44$
  - $c_{S2,C1} = 1732.65$, $c_{S2,C2} = 1780.72$, $c_{S2,C3} = 567.44$
  - $c_{S3,C1} = 115.66$, $c_{S3,C2} = 100.76$, $c_{S3,C3} = 64.68$

- $M = \sum_{j \in J} d_j = 1083 + 776 + 16214 = 18073$

##### Objective Function

\[
\min \left(
    1506.22\,x_{S1,C1} + 70.9\,x_{S1,C2} + 8.44\,x_{S1,C3}
  + 1732.65\,x_{S2,C1} + 1780.72\,x_{S2,C2} + 567.44\,x_{S2,C3}
  + 115.66\,x_{S3,C1} + 100.76\,x_{S3,C2} + 64.68\,x_{S3,C3}
  + 102.33\,y_{S1} + 94.92\,y_{S2} + 91.83\,y_{S3}
\right)
\]

##### Constraints

1. **Demand satisfaction for each musician/band:**
   \[
   x_{S1,j} + x_{S2,j} + x_{S3,j} = d_j, \quad \forall j \in \{C1, C2, C3\}
   \]
   That is:
   - $x_{S1,C1} + x_{S2,C1} + x_{S3,C1} = 1083$
   - $x_{S1,C2} + x_{S2,C2} + x_{S3,C2} = 776$
   - $x_{S1,C3} + x_{S2,C3} + x_{S3,C3} = 16214$

2. **Warehouse activation:**
   \[
   x_{i,C1} + x_{i,C2} + x_{i,C3} \leq M\,y_i, \quad \forall i \in \{S1, S2, S3\}
   \]
   Where $M = 18073$.

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Retrieved Information

{
  "warehouses": ["S1", "S2", "S3"],
  "musicians_bands": ["C1", "C2", "C3"],
  "demand": {
    "C1": 1083,
    "C2": 776,
    "C3": 16214
  },
  "fixed_cost": {
    "S1": 102.33,
    "S2": 94.92,
    "S3": 91.83
  },
  "cost": {
    "S1": {"C1": 1506.22, "C2": 70.9, "C3": 8.44},
    "S2": {"C1": 1732.65, "C2": 1780.72, "C3": 567.44},
    "S3": {"C1": 115.66, "C2": 100.76, "C3": 64.68}
  },
  "M": 18073
}