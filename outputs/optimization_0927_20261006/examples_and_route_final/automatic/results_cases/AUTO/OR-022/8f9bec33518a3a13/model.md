Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the ‘27in’ products) to be fulfilled.

##### Sets and Parameters

- Products:
  - Product 1: 27in 4K Gaming Monitor
  - Product 2: 27in FHD Monitor

- Revenue per unit:
  - $r_1 = 261.2933$ (27in 4K Gaming Monitor)
  - $r_2 = 52.4965$ (27in FHD Monitor)

- Demand:
  - $d_1 = 12474$ (27in 4K Gaming Monitor)
  - $d_2 = 15057$ (27in FHD Monitor)

- Initial Inventory:
  - $s_1 = 62440$ (27in 4K Gaming Monitor)
  - $s_2 = 75500$ (27in FHD Monitor)

##### Decision Variables

- $x_1$: Number of 27in 4K Gaming Monitors fulfilled (integer, $\geq 0$)
- $x_2$: Number of 27in FHD Monitors fulfilled (integer, $\geq 0$)

##### Objective Function

\[
\max \; 261.2933\, x_1 + 52.4965\, x_2
\]

##### Constraints

\[
\begin{align*}
& x_1 \leq 12474 \\
& x_1 \leq 62440 \\
& x_2 \leq 15057 \\
& x_2 \leq 75500 \\
& x_1, x_2 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

##### Complete Model

\[
\begin{align*}
\max \quad & 261.2933\, x_1 + 52.4965\, x_2 \\
\text{s.t.} \quad
& x_1 \leq 12474 \\
& x_1 \leq 62440 \\
& x_2 \leq 15057 \\
& x_2 \leq 75500 \\
& x_1, x_2 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_1$ = units of 27in 4K Gaming Monitor fulfilled
- $x_2$ = units of 27in FHD Monitor fulfilled