Let $x_i$ denote the number of units of product $i$ (here, "Baby Food_255.28") to be fulfilled.

Objective:
$$
\max\ 255.28\, x_i
$$

Subject to:
\begin{align*}
x_i &\leq 765850 \quad \text{(Demand constraint)} \\
x_i &\leq 5627060 \quad \text{(Initial Inventory constraint)} \\
x_i &\geq 0 \\
x_i &\in \mathbb{Z}
\end{align*}

Where:
- $x_i$ = number of units of "Baby Food_255.28" fulfilled (decision variable, nonnegative integer)
- 255.28 = Revenue per unit
- 765850 = Demand
- 5627060 = Initial Inventory