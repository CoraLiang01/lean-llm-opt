Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of ‘27in’ products) to be fulfilled.

Products (in source order):

- Product 1: 27in 4K Gaming Monitor
- Product 2: 27in FHD Monitor

Parameters:

- $r_i$: Revenue per unit of product $i$
- $d_i$: Demand for product $i$
- $s_i$: Initial Inventory of product $i$

Data:

\[
\begin{array}{lccc}
\text{Product Name} & r_i & d_i & s_i \\
\hline
\text{27in 4K Gaming Monitor} & 261.2933 & 12474 & 62440 \\
\text{27in FHD Monitor} & 52.4965 & 15057 & 75500 \\
\end{array}
\]

Model:

Objective:
\[
\max\ 261.2933\, x_1 + 52.4965\, x_2
\]

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
- $x_1$ = units of 27in 4K Gaming Monitor fulfilled
- $x_2$ = units of 27in FHD Monitor fulfilled

Each $x_i$ is a nonnegative integer, cannot exceed either demand or initial inventory for that product, and the objective is to maximize total revenue.