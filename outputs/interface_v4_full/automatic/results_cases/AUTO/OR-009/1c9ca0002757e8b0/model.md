##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from plant $i$ to retail outlet $j$, for all $i \in \{S1, S2, S3, S4\}$ and $j \in \{C1, C2, C3, C4\}$.

##### Objective Function

Minimize the total transportation cost:
$$
\min \Bigg[
543.756480860856\,x_{S1,C1} + 23.685276141764653\,x_{S1,C2} + 23.676386730773032\,x_{S1,C3} + 447.75143678673766\,x_{S1,C4} \\
+ 883.9151090405642\,x_{S2,C1} + 0.04977684765576961\,x_{S2,C2} + 0.0350986687216299\,x_{S2,C3} + 44.45588531711622\,x_{S2,C4} \\
+ 537.3456896658107\,x_{S3,C1} + 23.769274659075112\,x_{S3,C2} + 498.95659249465467\,x_{S3,C3} + 440.60737890439776\,x_{S3,C4} \\
+ 1791.493192397229\,x_{S4,C1} + 68.21633865655126\,x_{S4,C2} + 1432.4837339656747\,x_{S4,C3} + 1527.7635425462734\,x_{S4,C4}
\Bigg]
$$

##### Constraints

**Demand satisfaction (each outlet receives at least its demand):**
\[
\begin{align*}
x_{S1,C1} + x_{S2,C1} + x_{S3,C1} + x_{S4,C1} &\geq 94 \\
x_{S1,C2} + x_{S2,C2} + x_{S3,C2} + x_{S4,C2} &\geq 39 \\
x_{S1,C3} + x_{S2,C3} + x_{S3,C3} + x_{S4,C3} &\geq 65 \\
x_{S1,C4} + x_{S2,C4} + x_{S3,C4} + x_{S4,C4} &\geq 435 \\
\end{align*}
\]

**Supply capacity (no plant exceeds its production capacity):**
\[
\begin{align*}
x_{S1,C1} + x_{S1,C2} + x_{S1,C3} + x_{S1,C4} &\leq 2531 \\
x_{S2,C1} + x_{S2,C2} + x_{S2,C3} + x_{S2,C4} &\leq 20 \\
x_{S3,C1} + x_{S3,C2} + x_{S3,C3} + x_{S3,C4} &\leq 210 \\
x_{S4,C1} + x_{S4,C2} + x_{S4,C3} + x_{S4,C4} &\leq 241 \\
\end{align*}
\]

**Non-negativity:**
\[
x_{ij} \geq 0 \quad \forall i \in \{S1, S2, S3, S4\},\ j \in \{C1, C2, C3, C4\}
\]

##### Parameters (from retrieved data)

- Plants: $S1$, $S2$, $S3$, $S4$
- Retail outlets: $C1$, $C2$, $C3$, $C4$
- Demands: $d_{C1}=94$, $d_{C2}=39$, $d_{C3}=65$, $d_{C4}=435$
- Supply capacities: $s_{S1}=2531$, $s_{S2}=20$, $s_{S3}=210$, $s_{S4}=241$
- Transportation costs $c_{ij}$ as given in the objective function above.