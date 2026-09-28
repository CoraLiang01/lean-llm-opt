Let $S = \{\text{S1}, \text{S2}, \ldots, \text{S10}\}$ be the set of source locations, and $D = \{\text{D1}, \text{D2}, \ldots, \text{D20}\}$ the set of demand locations.

Parameters (from the retrieved data):

- Supply at each source $i \in S$:
  - $\text{supply}_{\text{S1}} = 103$
  - $\text{supply}_{\text{S2}} = 87$
  - $\text{supply}_{\text{S3}} = 95$
  - $\text{supply}_{\text{S4}} = 112$
  - $\text{supply}_{\text{S5}} = 97$
  - $\text{supply}_{\text{S6}} = 103$
  - $\text{supply}_{\text{S7}} = 101$
  - $\text{supply}_{\text{S8}} = 94$
  - $\text{supply}_{\text{S9}} = 102$
  - $\text{supply}_{\text{S10}} = 106$

- Demand at each destination $j \in D$:
  - $\text{demand}_{\text{D1}} = 61$
  - $\text{demand}_{\text{D2}} = 54$
  - $\text{demand}_{\text{D3}} = 56$
  - $\text{demand}_{\text{D4}} = 54$
  - $\text{demand}_{\text{D5}} = 53$
  - $\text{demand}_{\text{D6}} = 47$
  - $\text{demand}_{\text{D7}} = 56$
  - $\text{demand}_{\text{D8}} = 57$
  - $\text{demand}_{\text{D9}} = 56$
  - $\text{demand}_{\text{D10}} = 34$
  - $\text{demand}_{\text{D11}} = 55$
  - $\text{demand}_{\text{D12}} = 53$
  - $\text{demand}_{\text{D13}} = 37$
  - $\text{demand}_{\text{D14}} = 31$
  - $\text{demand}_{\text{D15}} = 62$
  - $\text{demand}_{\text{D16}} = 58$
  - $\text{demand}_{\text{D17}} = 39$
  - $\text{demand}_{\text{D18}} = 32$
  - $\text{demand}_{\text{D19}} = 38$
  - $\text{demand}_{\text{D20}} = 67$

- Unit transportation costs $c_{ij}$ (cost per unit shipped from source $i$ to destination $j$):

|        | D1   | D2   | D3   | D4   | D5   | D6   | D7   | D8   | D9   | D10  | D11  | D12  | D13  | D14  | D15  | D16  | D17  | D18  | D19  | D20  |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| S1     | 2.75 | 2.6  | 2.9  | 1.7  | 1.85 | 1.98 | 2.74 | 6.2  | 5.75 | 6.44 | 5.2  | 4.54 | 5.39 | 4.34 | 8.28 | 8.87 | 9.03 | 8.49 | 9.66 | 10.91|
| S2     | 5.24 | 4.87 | 4.72 | 4.12 | 4.26 | 4.55 | 4.45 | 3.88 | 3.24 | 4.24 | 3.13 | 3.16 | 3.67 | 2.43 | 5.83 | 6.51 | 6.68 | 6.1  | 7.2  | 8.4  |
| S3     | 5.54 | 5.16 | 5.09 | 4.37 | 4.57 | 4.78 | 4.89 | 3.84 | 3.08 | 4.42 | 3.37 | 3.53 | 3.98 | 2.7  | 5.84 | 6.48 | 6.68 | 6.14 | 7.24 | 8.31 |
| S4     | 4.51 | 4.17 | 4.01 | 3.36 | 3.59 | 3.73 | 3.68 | 4.39 | 3.93 | 4.71 | 3.55 | 3.15 | 3.84 | 2.6  | 6.51 | 7.03 | 7.27 | 6.69 | 7.89 | 9.04 |
| S5     | 4.84 | 4.74 | 4.77 | 3.8  | 3.96 | 4.1  | 4.52 | 4.82 | 4.03 | 5.35 | 4.25 | 4.14 | 4.73 | 3.46 | 6.89 | 7.5  | 7.67 | 7.2  | 8.2  | 9.35 |
| S6     | 9.13 | 8.32 | 7.78 | 7.9  | 8.13 | 8.45 | 7.47 | 3.04 | 4.25 | 2.19 | 3.34 | 4.07 | 3.3  | 4.24 | 2.77 | 2.63 | 2.97 | 2.56 | 3.32 | 5.09 |
| S7     | 8.64 | 7.86 | 7.34 | 7.32 | 7.51 | 7.83 | 7.03 | 1.9  | 3.14 | 1.43 | 2.64 | 3.64 | 2.92 | 3.51 | 2.41 | 2.74 | 2.98 | 2.46 | 3.51 | 5.08 |
| S8     | 9.25 | 8.41 | 7.81 | 8.13 | 8.32 | 8.6  | 7.49 | 3.71 | 5.0  | 2.88 | 3.78 | 4.27 | 3.49 | 4.63 | 3.62 | 3.34 | 3.59 | 3.29 | 3.79 | 5.54 |
| S9     |10.3  | 9.58 | 8.89 | 9.11 | 9.32 | 9.69 | 8.57 | 4.11 | 5.29 | 3.49 | 4.59 | 5.28 | 4.47 | 5.51 | 3.22 | 2.59 | 2.72 | 2.68 | 2.82 | 4.47 |
| S10    | 7.85 | 7.08 | 6.57 | 6.72 | 6.88 | 7.18 | 6.21 | 2.34 | 3.56 | 1.52 | 2.18 | 2.78 | 2.01 | 3.0  | 3.47 | 3.62 | 3.89 | 3.41 | 4.38 | 5.98 |

Decision variables:

- $y_{ij} \geq 0$: Amount of cargo (units) shipped from source $i$ to destination $j$ (continuous, can be fractional).
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of trucks dispatched from source $i$ to destination $j$ (integer, whole trucks).

Truck capacity: Each truck can carry at most 10 units.

Model:

Minimize total transportation cost:
$$
\min \sum_{i \in S} \sum_{j \in D} c_{ij} \cdot y_{ij}
$$

Subject to:

1. Supply constraints (do not exceed supply at each source):
$$
\sum_{j \in D} y_{ij} \leq \text{supply}_i, \quad \forall i \in S
$$

2. Demand constraints (meet demand at each destination):
$$
\sum_{i \in S} y_{ij} = \text{demand}_j, \quad \forall j \in D
$$

3. Truck loading constraints (cargo per route cannot exceed total truck capacity dispatched):
$$
y_{ij} \leq 10 \cdot x_{ij}, \quad \forall i \in S, \forall j \in D
$$

4. Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}, \quad y_{ij} \geq 0, \quad \forall i \in S, \forall j \in D
$$

All parameters and coefficients are as retrieved above, with $c_{ij}$, $\text{supply}_i$, and $\text{demand}_j$ as listed.

This model ensures:
- All demands are met exactly.
- No source ships more than its available supply.
- Each route uses only whole trucks, but truck loads can be partial (up to 10 units per truck).
- Costs are per unit shipped, as specified.