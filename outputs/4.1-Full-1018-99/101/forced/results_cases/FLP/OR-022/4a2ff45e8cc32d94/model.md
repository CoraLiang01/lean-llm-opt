##### Decision Variables

$x_i \geq 0$, integer: Number of units of product $i$ (where $i$ is a ‘27in’ product) to fulfill.

##### Parameters

Let $I$ be the set of ‘27in’ products:

- $I = \{\text{27in 4K Gaming Monitor},\ \text{27in FHD Monitor}\}$

For each $i \in I$:

- $\text{Revenue}_i$ = revenue per unit of product $i$
- $\text{Demand}_i$ = demand for product $i$
- $\text{Inventory}_i$ = initial inventory for product $i$

From the data:

| Product Name              | Revenue   | Demand | Initial Inventory |
|--------------------------|-----------|--------|------------------|
| 27in 4K Gaming Monitor   | 261.2933  | 12474  | 62440            |
| 27in FHD Monitor         | 52.4965   | 15057  | 75500            |

##### Objective Function

\[
\max \sum_{i \in I} \text{Revenue}_i \cdot x_i
\]
That is,
\[
\max\ 261.2933\, x_1 + 52.4965\, x_2
\]
where $x_1$ is the fulfilled quantity of 27in 4K Gaming Monitor, $x_2$ is the fulfilled quantity of 27in FHD Monitor.

##### Constraints

1. Inventory and demand limits:
   \[
   0 \leq x_i \leq \min(\text{Demand}_i,\, \text{Inventory}_i),\quad \forall i \in I
   \]
   Specifically:
   \[
   0 \leq x_1 \leq 12474
   \]
   \[
   0 \leq x_2 \leq 15057
   \]

2. Integer requirements:
   \[
   x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\max\quad & 261.2933\, x_1 + 52.4965\, x_2 \\
\text{s.t.}\quad
& 0 \leq x_1 \leq 12474 \\
& 0 \leq x_2 \leq 15057 \\
& x_1,\, x_2 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_1$ = units of 27in 4K Gaming Monitor fulfilled
- $x_2$ = units of 27in FHD Monitor fulfilled

All parameters are directly from the CSV data.