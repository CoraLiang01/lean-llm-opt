Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the ‘27in’ products below) to fulfill.

##### Sets and Parameters

- Products:
  1. 27in 4K Gaming Monitor
  2. 27in FHD Monitor

- For each product $i$:
    - $r_i$: Revenue per unit
    - $d_i$: Demand
    - $s_i$: Initial Inventory

| $i$ | Product Name              | $r_i$ (Revenue) | $d_i$ (Demand) | $s_i$ (Initial Inventory) |
|-----|--------------------------|-----------------|---------------|--------------------------|
| 1   | 27in 4K Gaming Monitor   | 389.99          | 12474         | 62440                    |
| 2   | 27in FHD Monitor         | 149.99          | 15057         | 75500                    |

##### Decision Variables

- $x_i \in \mathbb{Z}_{\geq 0}$, for each product $i$ (number of units of product $i$ to fulfill)

##### Objective

\[
\max \; 389.99\, x_1 + 149.99\, x_2
\]

##### Constraints

1. Demand fulfillment cannot exceed demand:
   \[
   x_1 \leq 12474
   \]
   \[
   x_2 \leq 15057
   \]

2. Fulfillment cannot exceed initial inventory:
   \[
   x_1 \leq 62440
   \]
   \[
   x_2 \leq 75500
   \]

3. Nonnegativity and integrality:
   \[
   x_1, x_2 \in \mathbb{Z}_{\geq 0}
   \]

##### Complete Model

\[
\begin{align*}
\max \quad & 389.99\, x_1 + 149.99\, x_2 \\
\text{s.t.} \quad
& x_1 \leq 12474 \\
& x_1 \leq 62440 \\
& x_2 \leq 15057 \\
& x_2 \leq 75500 \\
& x_1, x_2 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

where:
- $x_1$ = units of 27in 4K Gaming Monitor to fulfill
- $x_2$ = units of 27in FHD Monitor to fulfill