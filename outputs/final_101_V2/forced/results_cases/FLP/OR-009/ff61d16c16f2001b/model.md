##### Decision Variables

$x_{ij} \geq 0$: quantity of beverages shipped from facility $i \in I$ to customer $j \in J$ (continuous).

##### Parameters

- $I = \{S1, S2, S3, S4\}$ (production facilities)
- $J = \{C1, C2, C3, C4\}$ (retail outlets)

- Customer demands:
  - $d_{C1} = 94$
  - $d_{C2} = 39$
  - $d_{C3} = 65$
  - $d_{C4} = 435$

- Facility supply capacities:
  - $u_{S1} = 2531$
  - $u_{S2} = 20$
  - $u_{S3} = 210$
  - $u_{S4} = 241$

- Transportation costs per unit ($c_{ij}$):

|        | C1              | C2                | C3                | C4                |
|--------|-----------------|-------------------|-------------------|-------------------|
| S1     | 543.756480860856    | 23.685276141764653  | 23.676386730773032  | 447.75143678673766  |
| S2     | 883.9151090405642   | 0.04977684765576961 | 0.0350986687216299  | 44.45588531711622   |
| S3     | 537.3456896658107   | 23.769274659075112  | 498.95659249465467  | 440.60737890439776  |
| S4     | 1791.493192397229   | 68.21633865655126   | 1432.4837339656747  | 1527.7635425462734  |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   That is,
   - $\sum_{i \in I} x_{i,C1} = 94$
   - $\sum_{i \in I} x_{i,C2} = 39$
   - $\sum_{i \in I} x_{i,C3} = 65$
   - $\sum_{i \in I} x_{i,C4} = 435$

2. **Facility capacity:**  
   For each facility $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq u_i
   \]
   That is,
   - $\sum_{j \in J} x_{S1,j} \leq 2531$
   - $\sum_{j \in J} x_{S2,j} \leq 20$
   - $\sum_{j \in J} x_{S3,j} \leq 210$
   - $\sum_{j \in J} x_{S4,j} \leq 241$

3. **Nonnegativity:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Complete Mathematical Model

\[
\begin{align*}
\min\quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq u_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\end{align*}
\]

Where all parameters and sets are as specified above, with all values and identifiers preserved from the original data.