Let $x_{mp}$ be a binary variable equal to 1 if manager $m$ is assigned to project $p$, and 0 otherwise.

Managers: MA, MB, MC  
Projects: P1, P2, P3

Costs:
\[
\begin{array}{c|ccc}
 & \text{P1} & \text{P2} & \text{P3} \\
\hline
\text{MA} & 3000 & 3200 & 3100 \\
\text{MB} & 2800 & 3300 & 2900 \\
\text{MC} & 2900 & 3100 & 3000 \\
\end{array}
\]

Objective:
\[
\min \quad 3000\,x_{\text{MA},\text{P1}} + 3200\,x_{\text{MA},\text{P2}} + 3100\,x_{\text{MA},\text{P3}}
+ 2800\,x_{\text{MB},\text{P1}} + 3300\,x_{\text{MB},\text{P2}} + 2900\,x_{\text{MB},\text{P3}}
+ 2900\,x_{\text{MC},\text{P1}} + 3100\,x_{\text{MC},\text{P2}} + 3000\,x_{\text{MC},\text{P3}}
\]

Subject to:

Each manager is assigned to exactly one project:
\[
x_{\text{MA},\text{P1}} + x_{\text{MA},\text{P2}} + x_{\text{MA},\text{P3}} = 1
\]
\[
x_{\text{MB},\text{P1}} + x_{\text{MB},\text{P2}} + x_{\text{MB},\text{P3}} = 1
\]
\[
x_{\text{MC},\text{P1}} + x_{\text{MC},\text{P2}} + x_{\text{MC},\text{P3}} = 1
\]

Each project is assigned to exactly one manager:
\[
x_{\text{MA},\text{P1}} + x_{\text{MB},\text{P1}} + x_{\text{MC},\text{P1}} = 1
\]
\[
x_{\text{MA},\text{P2}} + x_{\text{MB},\text{P2}} + x_{\text{MC},\text{P2}} = 1
\]
\[
x_{\text{MA},\text{P3}} + x_{\text{MB},\text{P3}} + x_{\text{MC},\text{P3}} = 1
\]

Binary assignment variables:
\[
x_{mp} \in \{0,1\} \quad \forall m \in \{\text{MA},\text{MB},\text{MC}\},\ p \in \{\text{P1},\text{P2},\text{P3}\}
\]