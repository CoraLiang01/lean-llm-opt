Let $x_i$ be the number of units of each ‘27in’ product $i$ to fulfill, where $i$ indexes the following products in the order given:

1. 27in 4K Gaming Monitor
2. 27in FHD Monitor

Parameters (from the data):

- Revenue per unit: $r_1 = 389.99$, $r_2 = 149.99$
- Demand: $d_1 = 12474$, $d_2 = 15057$
- Initial Inventory: $s_1 = 62440$, $s_2 = 75500$

Mathematical Model:

Objective:
$$
\max\ 389.99\, x_1 + 149.99\, x_2
$$

Subject to:
\[
\begin{align*}
& x_1 \leq 12474 \\
& x_2 \leq 15057 \\
& x_1 \leq 62440 \\
& x_2 \leq 75500 \\
& x_1 \geq 0,\quad x_2 \geq 0 \\
& x_1,\, x_2 \in \mathbb{Z}
\end{align*}
\]

Where:
- $x_1$ = units of 27in 4K Gaming Monitor to fulfill
- $x_2$ = units of 27in FHD Monitor to fulfill

All variables are nonnegative integers, bounded above by both demand and initial inventory for each product. The objective is to maximize total revenue from fulfilling these products.