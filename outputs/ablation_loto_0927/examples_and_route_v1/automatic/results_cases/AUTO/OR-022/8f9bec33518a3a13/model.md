Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of ‘27in’ products) to be fulfilled.

##### Sets and Parameters

- Products (in source order):
  1. 27in 4K Gaming Monitor
  2. 27in FHD Monitor

- For each product $i$:
  - $r_i$: Revenue per unit
  - $d_i$: Demand
  - $s_i$: Initial Inventory

| Product Name              | $r_i$    | $d_i$  | $s_i$  |
|--------------------------|----------|--------|--------|
| 27in 4K Gaming Monitor   | 261.2933 | 12474  | 62440  |
| 27in FHD Monitor         | 52.4965  | 15057  | 75500  |

##### Decision Variables

- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to fulfill

##### Objective

$\max \quad 261.2933\, x_1 + 52.4965\, x_2$

##### Constraints

1. Demand constraints:
   $$
   x_1 \leq 12474
   $$
   $$
   x_2 \leq 15057
   $$

2. Inventory constraints:
   $$
   x_1 \leq 62440
   $$
   $$
   x_2 \leq 75500
   $$

3. Non-negativity and integrality:
   $$
   x_1,\, x_2 \in \mathbb{Z}_{\geq 0}
   $$

##### Complete Model

\[
\begin{align*}
\max \quad & 261.2933\, x_1 + 52.4965\, x_2 \\
\text{s.t.} \quad
& x_1 \leq 12474 \\
& x_2 \leq 15057 \\
& x_1 \leq 62440 \\
& x_2 \leq 75500 \\
& x_1,\, x_2 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_1$ = units of 27in 4K Gaming Monitor fulfilled
- $x_2$ = units of 27in FHD Monitor fulfilled