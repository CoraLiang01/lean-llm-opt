##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from plant $i$ to retail outlet $j$, for all $i \in I$ (plants) and $j \in J$ (retail outlets).

##### Parameters

Plants (suppliers): $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$

Retail outlets (customers): $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$

Demands:
\[
\begin{align*}
d_{\text{C1}} &= 94 \\
d_{\text{C2}} &= 39 \\
d_{\text{C3}} &= 65 \\
d_{\text{C4}} &= 435 \\
\end{align*}
\]

Supply capacities:
\[
\begin{align*}
s_{\text{S1}} &= 2531 \\
s_{\text{S2}} &= 20 \\
s_{\text{S3}} &= 210 \\
s_{\text{S4}} &= 241 \\
\end{align*}
\]

Transportation costs:
\[
\begin{array}{c|cccc}
 & \text{C1} & \text{C2} & \text{C3} & \text{C4} \\
\hline
\text{S1} & 543.756480860856 & 23.685276141764653 & 23.676386730773032 & 447.75143678673766 \\
\text{S2} & 883.9151090405642 & 0.04977684765576961 & 0.0350986687216299 & 44.45588531711622 \\
\text{S3} & 537.3456896658107 & 23.769274659075112 & 498.95659249465467 & 440.60737890439776 \\
\text{S4} & 1791.493192397229 & 68.21633865655126 & 1432.4837339656747 & 1527.7635425462734 \\
\end{array}
\]

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from plant $i$ to outlet $j$ (see table above).

##### Constraints

1. Demand satisfaction (each outlet receives at least its demand):
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
   That is,
   \begin{align*}
   x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} + x_{\text{S4},\text{C1}} &\geq 94 \\
   x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} + x_{\text{S4},\text{C2}} &\geq 39 \\
   x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} + x_{\text{S4},\text{C3}} &\geq 65 \\
   x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + x_{\text{S3},\text{C4}} + x_{\text{S4},\text{C4}} &\geq 435 \\
   \end{align*}

2. Supply capacity (no plant ships more than its capacity):
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
   That is,
   \begin{align*}
   x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} + x_{\text{S1},\text{C4}} &\leq 2531 \\
   x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} + x_{\text{S2},\text{C4}} &\leq 20 \\
   x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} + x_{\text{S3},\text{C4}} &\leq 210 \\
   x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + x_{\text{S4},\text{C3}} + x_{\text{S4},\text{C4}} &\leq 241 \\
   \end{align*}

3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Complete Model

\[
\begin{align*}
\min\ & 543.756480860856\,x_{\text{S1},\text{C1}} + 23.685276141764653\,x_{\text{S1},\text{C2}} + 23.676386730773032\,x_{\text{S1},\text{C3}} + 447.75143678673766\,x_{\text{S1},\text{C4}} \\
&+ 883.9151090405642\,x_{\text{S2},\text{C1}} + 0.04977684765576961\,x_{\text{S2},\text{C2}} + 0.0350986687216299\,x_{\text{S2},\text{C3}} + 44.45588531711622\,x_{\text{S2},\text{C4}} \\
&+ 537.3456896658107\,x_{\text{S3},\text{C1}} + 23.769274659075112\,x_{\text{S3},\text{C2}} + 498.95659249465467\,x_{\text{S3},\text{C3}} + 440.60737890439776\,x_{\text{S3},\text{C4}} \\
&+ 1791.493192397229\,x_{\text{S4},\text{C1}} + 68.21633865655126\,x_{\text{S4},\text{C2}} + 1432.4837339656747\,x_{\text{S4},\text{C3}} + 1527.7635425462734\,x_{\text{S4},\text{C4}} \\
\text{s.t.}\quad
& x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} + x_{\text{S4},\text{C1}} \geq 94 \\
& x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} + x_{\text{S4},\text{C2}} \geq 39 \\
& x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} + x_{\text{S4},\text{C3}} \geq 65 \\
& x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + x_{\text{S3},\text{C4}} + x_{\text{S4},\text{C4}} \geq 435 \\
& x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} + x_{\text{S1},\text{C4}} \leq 2531 \\
& x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} + x_{\text{S2},\text{C4}} \leq 20 \\
& x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} + x_{\text{S3},\text{C4}} \leq 210 \\
& x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + x_{\text{S4},\text{C3}} + x_{\text{S4},\text{C4}} \leq 241 \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J \\
\end{align*}
\]