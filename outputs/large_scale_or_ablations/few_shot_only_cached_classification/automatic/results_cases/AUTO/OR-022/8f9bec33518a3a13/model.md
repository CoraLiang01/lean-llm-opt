Let $i$ index all products classified as ‘27in’ in the source order:

- $i=1$: 27in 4K Gaming Monitor
- $i=2$: 27in FHD Monitor

Let $x_i$ be the number of units of product $i$ to fulfill.

Parameters (from the retrieved data):

\[
\begin{align*}
&\text{Product 1:} \quad \text{27in 4K Gaming Monitor} \\
&\quad \text{Revenue:} \quad 261.2933 \\
&\quad \text{Demand:} \quad 12474 \\
&\quad \text{Initial Inventory:} \quad 62440 \\[1em]
&\text{Product 2:} \quad \text{27in FHD Monitor} \\
&\quad \text{Revenue:} \quad 52.4965 \\
&\quad \text{Demand:} \quad 15057 \\
&\quad \text{Initial Inventory:} \quad 75500 \\
\end{align*}
\]

The complete mathematical model is:

Objective:
\[
\max \quad 261.2933\, x_1 + 52.4965\, x_2
\]

Subject to:
\[
\begin{align*}
&x_1 \leq 12474 \\
&x_1 \leq 62440 \\
&x_2 \leq 15057 \\
&x_2 \leq 75500 \\
&x_1 \geq 0,\quad x_2 \geq 0 \\
&x_1,\, x_2 \in \mathbb{Z}
\end{align*}
\]

Where:
- $x_1$ = units of 27in 4K Gaming Monitor fulfilled
- $x_2$ = units of 27in FHD Monitor fulfilled

All coefficients and identifiers are preserved in source order.