Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of ‘27in’ products) to fulfill.

Products:
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
\max \; 389.99\, x_1 + 149.99\, x_2
\]

Subject to:
\[
\begin{align*}
& x_1 \leq 12474 \\
& x_1 \leq 62440 \\
& x_2 \leq 15057 \\
& x_2 \leq 75500 \\
& x_1 \geq 0,\; x_1 \in \mathbb{Z} \\
& x_2 \geq 0,\; x_2 \in \mathbb{Z}
\end{align*}
\]

Or, equivalently (since demand is less than inventory for both products):
\[
\begin{align*}
& 0 \leq x_1 \leq 12474,\quad x_1 \in \mathbb{Z} \\
& 0 \leq x_2 \leq 15057,\quad x_2 \in \mathbb{Z}
\end{align*}
\]

Where:
- $x_1$: units of 27in 4K Gaming Monitor to fulfill
- $x_2$: units of 27in FHD Monitor to fulfill