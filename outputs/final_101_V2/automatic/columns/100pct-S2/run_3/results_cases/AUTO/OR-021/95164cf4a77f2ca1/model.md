##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from plant $i$ to retail outlet $j$, for all $i \in I$ (plants) and $j \in J$ (retail outlets).

##### Sets

- $I = \{S1, S2, S3, S4\}$ (production plants)
- $J = \{C1, C2, C3, C4\}$ (retail outlets)

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

##### Mathematical Model

Minimize total transportation cost:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from plant $i$ to outlet $j$ (see table above).

Subject to:

1. Demand satisfaction at each retail outlet:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
   (Each outlet receives at least its demand.)

2. Supply capacity at each plant:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
   (No plant ships more than its capacity.)

3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### All coefficients and identifiers (source order):

- Retail outlets: C1, C2, C3, C4
- Plants: S1, S2, S3, S4
- Demands: 94, 39, 65, 435
- Supply capacities: 2531, 20, 210, 241
- Transportation costs (by plant, then outlet):

  - S1: 543.756480860856 (C1), 23.685276141764653 (C2), 23.676386730773032 (C3), 447.75143678673766 (C4)
  - S2: 883.9151090405642 (C1), 0.04977684765576961 (C2), 0.0350986687216299 (C3), 44.45588531711622 (C4)
  - S3: 537.3456896658107 (C1), 23.769274659075112 (C2), 498.95659249465467 (C3), 440.60737890439776 (C4)
  - S4: 1791.493192397229 (C1), 68.21633865655126 (C2), 1432.4837339656747 (C3), 1527.7635425462734 (C4)