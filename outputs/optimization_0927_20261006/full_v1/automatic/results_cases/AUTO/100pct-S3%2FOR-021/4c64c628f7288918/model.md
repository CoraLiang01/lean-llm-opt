##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from plant $i$ to retail outlet $j$, for all $i \in I$ (plants) and $j \in J$ (retail outlets).

##### Sets

- $I = \{S1, S2, S3, S4\}$ (production plants)
- $J = \{C1, C2, C3, C4\}$ (retail outlets)

##### Parameters

- Demands:
  - $d_{C1} = 94$
  - $d_{C2} = 39$
  - $d_{C3} = 65$
  - $d_{C4} = 435$
- Supply capacities:
  - $s_{S1} = 2531$
  - $s_{S2} = 20$
  - $s_{S3} = 210$
  - $s_{S4} = 241$
- Transportation costs $c_{ij}$:

\[
\begin{array}{c|cccc}
 & C1 & C2 & C3 & C4 \\
\hline
S1 & 543.756480860856 & 23.685276141764653 & 23.676386730773032 & 447.75143678673766 \\
S2 & 883.9151090405642 & 0.04977684765576961 & 0.0350986687216299 & 44.45588531711622 \\
S3 & 537.3456896658107 & 23.769274659075112 & 498.95659249465467 & 440.60737890439776 \\
S4 & 1791.493192397229 & 68.21633865655126 & 1432.4837339656747 & 1527.7635425462734 \\
\end{array}
\]

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction** (each retail outlet receives at least its demand):

   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
   Specifically:
   \begin{align*}
   x_{S1,C1} + x_{S2,C1} + x_{S3,C1} + x_{S4,C1} &\geq 94 \\
   x_{S1,C2} + x_{S2,C2} + x_{S3,C2} + x_{S4,C2} &\geq 39 \\
   x_{S1,C3} + x_{S2,C3} + x_{S3,C3} + x_{S4,C3} &\geq 65 \\
   x_{S1,C4} + x_{S2,C4} + x_{S3,C4} + x_{S4,C4} &\geq 435 \\
   \end{align*}

2. **Supply capacity** (no plant ships more than its capacity):

   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
   Specifically:
   \begin{align*}
   x_{S1,C1} + x_{S1,C2} + x_{S1,C3} + x_{S1,C4} &\leq 2531 \\
   x_{S2,C1} + x_{S2,C2} + x_{S2,C3} + x_{S2,C4} &\leq 20 \\
   x_{S3,C1} + x_{S3,C2} + x_{S3,C3} + x_{S3,C4} &\leq 210 \\
   x_{S4,C1} + x_{S4,C2} + x_{S4,C3} + x_{S4,C4} &\leq 241 \\
   \end{align*}

3. **Non-negativity**:

   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Complete Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\end{align*}
\]

Where all parameters and indices are as specified above, with all coefficients and identifiers preserved from the retrieved data.