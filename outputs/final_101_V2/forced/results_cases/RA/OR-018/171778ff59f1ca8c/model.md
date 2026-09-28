Let $x_i$ be the number of units of Baby product $i$ to fulfill.

Parameters (from data):

- Product: Baby Food_255.28
- Revenue: $255.28$
- Demand: $3,\!066,\!513$
- Initial Inventory: $22,\!749,\!210$

Decision variable:
- $x_i \in \mathbb{Z}_{\geq 0}$ (number of units of Baby Food_255.28 to fulfill)

Objective:
$$
\max\ 255.28\, x_i
$$

Subject to:
\[
\begin{align*}
x_i &\leq 3,\!066,\!513 \quad &\text{(Demand constraint)} \\
x_i &\leq 22,\!749,\!210 \quad &\text{(Initial inventory constraint)} \\
x_i &\geq 0,\ x_i \in \mathbb{Z} \quad &\text{(Nonnegativity and integrality)}
\end{align*}
\]

Where:
- $x_i$ = units of Baby Food_255.28 fulfilled.