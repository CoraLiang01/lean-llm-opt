Let $x_i$ denote the number of units of each ‘27in’ product $i$ to fulfill, where $i$ indexes the following products in source order:

- $i=1$: 27in 4K Gaming Monitor
- $i=2$: 27in FHD Monitor

Parameters:
- $r_i$: Revenue per unit of product $i$
  - $r_1 = 389.99$
  - $r_2 = 149.99$
- $d_i$: Demand for product $i$
  - $d_1 = 12474$
  - $d_2 = 15057$
- $s_i$: Initial Inventory for product $i$
  - $s_1 = 62440$
  - $s_2 = 75500$

Decision variables:
- $x_i \in \mathbb{Z}_{\geq 0}$, for $i=1,2$

Objective:
\[
\max\ 389.99\, x_1 + 149.99\, x_2
\]

Subject to:
\[
\begin{align*}
& x_1 \leq 12474 \quad &\text{(Demand constraint for 27in 4K Gaming Monitor)} \\
& x_2 \leq 15057 \quad &\text{(Demand constraint for 27in FHD Monitor)} \\
& x_1 \leq 62440 \quad &\text{(Inventory constraint for 27in 4K Gaming Monitor)} \\
& x_2 \leq 75500 \quad &\text{(Inventory constraint for 27in FHD Monitor)} \\
& x_1,\, x_2 \geq 0,\quad x_1,\, x_2 \in \mathbb{Z}
\end{align*}
\]

Where:
- $x_1$ = units of 27in 4K Gaming Monitor to fulfill
- $x_2$ = units of 27in FHD Monitor to fulfill

All coefficients and identifiers are as retrieved and in source order.