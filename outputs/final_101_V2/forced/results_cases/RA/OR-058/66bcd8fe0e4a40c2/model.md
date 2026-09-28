Let $F = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}\}$ be the set of suppliers, and $C = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}\}$ the set of stores.

Let:
- $y_i \in \{0,1\}$ indicate if supplier $i \in F$ is operational.
- $x_{ij} \geq 0$ is the quantity supplied from supplier $i \in F$ to store $j \in C$.

Parameters:
- Fixed costs for each supplier $i$:
  - S1: $98.88$
  - S2: $99.73$
  - S3: $94.01$
  - S4: $93.77$
  - S5: $107.59$
  - S6: $112.65$
- Transportation costs per unit from supplier $i$ to store $j$:

|        | C1     | C2     | C3      | C4      | C5     | C6     |
|--------|--------|--------|---------|---------|--------|--------|
| S1     | 0.08   | 52.33  | 73.57   | 1237.33 | 0.07   | 112.16 |
| S2     | 46.02  | 175.23 | 2026.83 | 299.89  | 966.53 | 1590.42|
| S3     |1031.74 | 78.13  | 99.02   | 277.07  | 884.45 | 1800.86|
| S4     | 868.75 | 94.2   |1776.34  | 285.48  | 868.85 | 86.55  |
| S5     |1577    | 760.15 |2090.19  | 43.2    |1577.12 |1095.17 |
| S6     | 49.14  | 4.33   |2079.57  | 277.04  |1032.01 |1543.49 |

- Demand at each store $j$:
  - C1: $216$
  - C2: $216$
  - C3: $216$
  - C4: $144$
  - C5: $144$
  - C6: $144$

Model:

Minimize total cost:
$$
\min \sum_{i \in F} \text{fixed\_costs}_i \cdot y_i + \sum_{i \in F} \sum_{j \in C} \text{transportation\_cost}_{ij} \cdot x_{ij}
$$

Where:
- $\text{fixed\_costs}_i$ is as above for each $i$.
- $\text{transportation\_cost}_{ij}$ is as in the table above.

Subject to:

1. Demand satisfaction at each store:
$$
\sum_{i \in F} x_{ij} = \text{demand}_j \qquad \forall j \in C
$$

2. Linking supplier activation to shipments:
$$
\sum_{j \in C} x_{ij} \leq M \cdot y_i \qquad \forall i \in F
$$
where $M$ is a sufficiently large constant (e.g., $M = \sum_j \text{demand}_j = 1080$).

3. Variable domains:
$$
x_{ij} \geq 0 \quad \text{and integer} \qquad \forall i \in F,\, j \in C
$$
$$
y_i \in \{0,1\} \qquad \forall i \in F
$$

All coefficients and identifiers are as retrieved above, in original order.