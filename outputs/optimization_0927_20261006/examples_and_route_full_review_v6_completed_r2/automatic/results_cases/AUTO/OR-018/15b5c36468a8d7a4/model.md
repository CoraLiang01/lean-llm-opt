Let $x_i$ denote the number of units of each Baby product $i$ to fulfill.

Parameters (from data):

- Product: Baby Food_255.28
- Revenue: $r_i = 255.28$
- Demand: $d_i = 3,\!066,\!513$
- Initial Inventory: $s_i = 22,\!749,\!210$

Decision variable:

- $x_i \in \mathbb{Z}_{\geq 0}$

Mathematical Model:

Objective:
$$
\max\ 255.28\, x_i
$$

Subject to:
$$
0 \leq x_i \leq 3,\!066,\!513 \\
x_i \leq 22,\!749,\!210 \\
x_i \in \mathbb{Z}_{\geq 0}
$$

Or, equivalently:
$$
0 \leq x_i \leq \min\{3,\!066,\!513,\ 22,\!749,\!210\} \\
x_i \in \mathbb{Z}_{\geq 0}
$$

Where $x_i$ is the number of units of "Baby Food_255.28" to fulfill.