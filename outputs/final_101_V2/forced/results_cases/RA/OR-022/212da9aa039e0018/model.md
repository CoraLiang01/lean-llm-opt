Let $x_i$ denote the number of units of each ‘27in’ product $i$ to be fulfilled.

Products (in source order):

- Product 1: 27in 4K Gaming Monitor
  - Revenue: $261.2933$
  - Demand: $12474$
  - Initial Inventory: $62440$
- Product 2: 27in FHD Monitor
  - Revenue: $52.4965$
  - Demand: $15057$
  - Initial Inventory: $75500$

Decision variables:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,2\}
$$

Objective:
$$
\max \; 261.2933\, x_1 + 52.4965\, x_2
$$

Subject to:
\[
\begin{align*}
& x_1 \leq 12474 \quad &\text{(Demand for 27in 4K Gaming Monitor)} \\
& x_1 \leq 62440 \quad &\text{(Initial Inventory for 27in 4K Gaming Monitor)} \\
& x_2 \leq 15057 \quad &\text{(Demand for 27in FHD Monitor)} \\
& x_2 \leq 75500 \quad &\text{(Initial Inventory for 27in FHD Monitor)} \\
& x_1, x_2 \geq 0 \text{ and integer}
\end{align*}
\]

Where:
- $x_1$ = units of 27in 4K Gaming Monitor fulfilled
- $x_2$ = units of 27in FHD Monitor fulfilled