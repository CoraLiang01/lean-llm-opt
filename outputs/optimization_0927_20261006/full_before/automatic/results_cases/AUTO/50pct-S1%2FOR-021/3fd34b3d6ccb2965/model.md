##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from plant $i$ to retail outlet $j$, for all $i \in I$ (plants) and $j \in J$ (retail outlets).

##### Parameters

- $I = \{S1, S2, S3, S4\}$ (plants)
- $J = \{C1, C2, C3, C4\}$ (retail outlets)

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

That is,

\[
\begin{align*}
\min\ &543.756480860856\,x_{S1,C1} + 23.685276141764653\,x_{S1,C2} + 23.676386730773032\,x_{S1,C3} + 447.75143678673766\,x_{S1,C4} \\
&+ 883.9151090405642\,x_{S2,C1} + 0.04977684765576961\,x_{S2,C2} + 0.0350986687216299\,x_{S2,C3} + 44.45588531711622\,x_{S2,C4} \\
&+ 537.3456896658107\,x_{S3,C1} + 23.769274659075112\,x_{S3,C2} + 498.95659249465467\,x_{S3,C3} + 440.60737890439776\,x_{S3,C4} \\
&+ 1791.493192397229\,x_{S4,C1} + 68.21633865655126\,x_{S4,C2} + 1432.4837339656747\,x_{S4,C3} + 1527.7635425462734\,x_{S4,C4}
\end{align*}
\]

##### Constraints

1. Demand satisfaction (each outlet receives at least its demand):

\[
\begin{align*}
x_{S1,C1} + x_{S2,C1} + x_{S3,C1} + x_{S4,C1} &\geq 94 \\
x_{S1,C2} + x_{S2,C2} + x_{S3,C2} + x_{S4,C2} &\geq 39 \\
x_{S1,C3} + x_{S2,C3} + x_{S3,C3} + x_{S4,C3} &\geq 65 \\
x_{S1,C4} + x_{S2,C4} + x_{S3,C4} + x_{S4,C4} &\geq 435 \\
\end{align*}
\]

2. Supply capacity (no plant ships more than its capacity):

\[
\begin{align*}
x_{S1,C1} + x_{S1,C2} + x_{S1,C3} + x_{S1,C4} &\leq 2531 \\
x_{S2,C1} + x_{S2,C2} + x_{S2,C3} + x_{S2,C4} &\leq 20 \\
x_{S3,C1} + x_{S3,C2} + x_{S3,C3} + x_{S3,C4} &\leq 210 \\
x_{S4,C1} + x_{S4,C2} + x_{S4,C3} + x_{S4,C4} &\leq 241 \\
\end{align*}
\]

3. Non-negativity:

\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]