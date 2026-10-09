##### Sets

- Plants (suppliers): $I = \{S1, S2, S3, S4\}$
- Retail outlets (customers): $J = \{C1, C2, C3, C4\}$

##### Parameters

- Demand at each retail outlet:
  - $d_{C1} = 94$
  - $d_{C2} = 39$
  - $d_{C3} = 65$
  - $d_{C4} = 435$
- Supply capacity at each plant:
  - $s_{S1} = 2531$
  - $s_{S2} = 20$
  - $s_{S3} = 210$
  - $s_{S4} = 241$
- Transportation costs per unit from each plant to each outlet:

|           | C1             | C2             | C3             | C4             |
|-----------|----------------|----------------|----------------|----------------|
| S1        | 543.756480860856   | 23.685276141764653  | 23.676386730773032  | 447.75143678673766  |
| S2        | 883.9151090405642  | 0.04977684765576961 | 0.0350986687216299  | 44.45588531711622   |
| S3        | 537.3456896658107  | 23.769274659075112  | 498.95659249465467  | 440.60737890439776  |
| S4        | 1791.493192397229  | 68.21633865655126   | 1432.4837339656747  | 1527.7635425462734  |

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to outlet $j \in J$ (continuous).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

where $c_{ij}$ is the transportation cost per unit from plant $i$ to outlet $j$ (see table above).

##### Constraints

1. **Demand satisfaction** (each outlet receives at least its demand):

   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
   - $\sum_{i \in I} x_{i,C1} \geq 94$
   - $\sum_{i \in I} x_{i,C2} \geq 39$
   - $\sum_{i \in I} x_{i,C3} \geq 65$
   - $\sum_{i \in I} x_{i,C4} \geq 435$

2. **Supply capacity** (no plant exceeds its capacity):

   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
   - $\sum_{j \in J} x_{S1,j} \leq 2531$
   - $\sum_{j \in J} x_{S2,j} \leq 20$
   - $\sum_{j \in J} x_{S3,j} \leq 210$
   - $\sum_{j \in J} x_{S4,j} \leq 241$

3. **Non-negativity**:

   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Complete Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\end{align*}
\]

where all parameters and indices are as defined above.