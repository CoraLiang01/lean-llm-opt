##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from plant $i$ to retail outlet $j$, for all $i \in I$ (plants) and $j \in J$ (retail outlets).

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (plants)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets)

##### Parameters

- Demands:
  - $d_{\text{C1}} = 94$
  - $d_{\text{C2}} = 39$
  - $d_{\text{C3}} = 65$
  - $d_{\text{C4}} = 435$
- Supply capacities:
  - $s_{\text{S1}} = 2531$
  - $s_{\text{S2}} = 20$
  - $s_{\text{S3}} = 210$
  - $s_{\text{S4}} = 241$
- Transportation costs $c_{ij}$:

|        | C1                | C2                | C3                | C4                |
|--------|-------------------|-------------------|-------------------|-------------------|
| S1     | 543.756480860856  | 23.685276141764653| 23.676386730773032| 447.75143678673766|
| S2     | 883.9151090405642 | 0.04977684765576961| 0.0350986687216299| 44.45588531711622 |
| S3     | 537.3456896658107 | 23.769274659075112| 498.95659249465467| 440.60737890439776|
| S4     | 1791.493192397229 | 68.21633865655126 | 1432.4837339656747| 1527.7635425462734|

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each outlet receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
   Explicitly:
   - $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} + x_{\text{S4},\text{C1}} \geq 94$
   - $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} + x_{\text{S4},\text{C2}} \geq 39$
   - $x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} + x_{\text{S4},\text{C3}} \geq 65$
   - $x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + x_{\text{S3},\text{C4}} + x_{\text{S4},\text{C4}} \geq 435$

2. **Supply capacity** (no plant exceeds its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
   Explicitly:
   - $x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} + x_{\text{S1},\text{C4}} \leq 2531$
   - $x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} + x_{\text{S2},\text{C4}} \leq 20$
   - $x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} + x_{\text{S3},\text{C4}} \leq 210$
   - $x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + x_{\text{S4},\text{C3}} + x_{\text{S4},\text{C4}} \leq 241$

3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Parameter Tables

**Demands**

| Customer | Demand |
|----------|--------|
| C1       | 94     |
| C2       | 39     |
| C3       | 65     |
| C4       | 435    |

**Supply Capacities**

| Plant | Supply Capacity |
|-------|----------------|
| S1    | 2531           |
| S2    | 20             |
| S3    | 210            |
| S4    | 241            |

**Transportation Costs**

|        | C1                | C2                | C3                | C4                |
|--------|-------------------|-------------------|-------------------|-------------------|
| S1     | 543.756480860856  | 23.685276141764653| 23.676386730773032| 447.75143678673766|
| S2     | 883.9151090405642 | 0.04977684765576961| 0.0350986687216299| 44.45588531711622 |
| S3     | 537.3456896658107 | 23.769274659075112| 498.95659249465467| 440.60737890439776|
| S4     | 1791.493192397229 | 68.21633865655126 | 1432.4837339656747| 1527.7635425462734|