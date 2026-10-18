##### Sets and Indices

- Warehouses (facilities): $i \in \{S1, S2, S3\}$
- Musicians/Bands (customers): $j \in \{C1, C2, C3\}$

##### Parameters

- Fixed costs for opening warehouse $i$:
  - $f_{S1} = 102.33$
  - $f_{S2} = 94.92$
  - $f_{S3} = 91.83$
- Demand for each customer $j$:
  - $d_{C1} = 1083$
  - $d_{C2} = 776$
  - $d_{C3} = 16214$
- Transportation cost per unit from warehouse $i$ to customer $j$:
  - $c_{S1,C1} = 1506.22$, $c_{S1,C2} = 70.90$, $c_{S1,C3} = 8.44$
  - $c_{S2,C1} = 1732.65$, $c_{S2,C2} = 1780.72$, $c_{S2,C3} = 567.44$
  - $c_{S3,C1} = 115.66$, $c_{S3,C2} = 100.76$, $c_{S3,C3} = 64.68$

##### Decision Variables

- $y_i \in \{0,1\}$: $y_i = 1$ if warehouse $i$ is opened, $0$ otherwise.
- $x_{ij} \geq 0$: Quantity of goods supplied from warehouse $i$ to customer $j$.

##### Objective Function

$$
\min \left( \sum_{i \in \{S1, S2, S3\}} f_i y_i + \sum_{i \in \{S1, S2, S3\}} \sum_{j \in \{C1, C2, C3\}} c_{ij} x_{ij} \right)
$$

##### Constraints

1. **Demand Satisfaction** (each customer's demand must be met):

   $$
   \sum_{i \in \{S1, S2, S3\}} x_{ij} = d_j \quad \forall j \in \{C1, C2, C3\}
   $$

2. **Warehouse Activation** (no supply from unopened warehouses):

   $$
   \sum_{j \in \{C1, C2, C3\}} x_{ij} \leq M_i y_i \quad \forall i \in \{S1, S2, S3\}
   $$

   Where $M_i = \sum_{j \in \{C1, C2, C3\}} d_j = 1083 + 776 + 16214 = 18073$ (a sufficiently large number to allow full supply if warehouse $i$ is open).

3. **Variable Domains**:

   $$
   y_i \in \{0,1\} \quad \forall i \in \{S1, S2, S3\}
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in \{S1, S2, S3\},\ j \in \{C1, C2, C3\}
   $$

##### Retrieved Information

{
  "fixed_costs": {
    "S1": 102.33,
    "S2": 94.92,
    "S3": 91.83
  },
  "demand": {
    "C1": 1083,
    "C2": 776,
    "C3": 16214
  },
  "transportation_costs": {
    "S1": {"C1": 1506.22, "C2": 70.90, "C3": 8.44},
    "S2": {"C1": 1732.65, "C2": 1780.72, "C3": 567.44},
    "S3": {"C1": 115.66, "C2": 100.76, "C3": 64.68}
  },
  "warehouses": ["S1", "S2", "S3"],
  "customers": ["C1", "C2", "C3"]
}