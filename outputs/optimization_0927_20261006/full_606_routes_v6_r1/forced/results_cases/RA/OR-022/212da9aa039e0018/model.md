Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of ‘27in’ products) to be fulfilled.

Parameters (from the retrieved data, in source order):

- Product 1: "27in 4K Gaming Monitor"
  - Revenue: $261.2933$
  - Demand: $12474$
  - Initial Inventory: $62440$
- Product 2: "27in FHD Monitor"
  - Revenue: $52.4965$
  - Demand: $15057$
  - Initial Inventory: $75500$

Mathematical Model:

Objective:
\[
\max \; 261.2933\, x_1 + 52.4965\, x_2
\]

Subject to:
\[
\begin{align*}
& x_1 \leq 12474 \quad &\text{(Demand constraint for 27in 4K Gaming Monitor)} \\
& x_1 \leq 62440 \quad &\text{(Inventory constraint for 27in 4K Gaming Monitor)} \\
& x_2 \leq 15057 \quad &\text{(Demand constraint for 27in FHD Monitor)} \\
& x_2 \leq 75500 \quad &\text{(Inventory constraint for 27in FHD Monitor)} \\
& x_1 \geq 0,\; x_2 \geq 0 \\
& x_1, x_2 \in \mathbb{Z} \\
\end{align*}
\]

Where:
- $x_1$ = units of "27in 4K Gaming Monitor" fulfilled
- $x_2$ = units of "27in FHD Monitor" fulfilled

All variables are nonnegative integers.