Let $x_i$ denote the number of units of Baby product $i$ to fulfill.

Parameters (from the data):

- Product: Baby Food_255.28
- Revenue: $255.28$
- Demand: $3,\!066,\!513$
- Initial Inventory: $22,\!749,\!210$

Decision variable:
- $x_1$: Number of units of Baby Food_255.28 to fulfill

Objective:
$$
\max\ 255.28\, x_1
$$

Subject to:
\[
\begin{align*}
& x_1 \leq 3,\!066,\!513 \quad \text{(Demand constraint)} \\
& x_1 \leq 22,\!749,\!210 \quad \text{(Initial Inventory constraint)} \\
& x_1 \geq 0,\quad x_1 \in \mathbb{Z} \quad \text{(Nonnegativity and integrality)}
\end{align*}
\]

Where:
- $x_1$ is the number of units of Baby Food_255.28 to fulfill.