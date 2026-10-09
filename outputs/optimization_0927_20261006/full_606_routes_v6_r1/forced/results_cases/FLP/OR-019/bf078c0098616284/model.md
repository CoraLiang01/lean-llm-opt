##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: Number of units of ‘27in’ product $i$ to fulfill, for each $i \in I$ (where $I$ is the set of ‘27in’ products).

##### Parameters

Let $I = \{$
- $1$: 27in 4K Gaming Monitor
- $2$: 27in FHD Monitor
$\}$

Product data:
- 27in 4K Gaming Monitor: Revenue = 389.99, Demand = 12474, Initial Inventory = 62440
- 27in FHD Monitor: Revenue = 149.99, Demand = 15057, Initial Inventory = 75500

Let:
- $r_i$: Revenue per unit of product $i$
- $d_i$: Demand for product $i$
- $s_i$: Initial inventory for product $i$

So,
- $r_1 = 389.99$, $d_1 = 12474$, $s_1 = 62440$
- $r_2 = 149.99$, $d_2 = 15057$, $s_2 = 75500$

##### Objective Function

\[
\max \; 389.99\,x_1 + 149.99\,x_2
\]

##### Constraints

1. Demand fulfillment: $x_i \leq d_i \quad \forall i \in I$
2. Inventory limit: $x_i \leq s_i \quad \forall i \in I$
3. Nonnegativity and integrality: $x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

##### Complete Model

\[
\begin{align*}
\max \quad & 389.99\,x_1 + 149.99\,x_2 \\
\text{s.t.} \quad & x_1 \leq 12474 \\
                  & x_2 \leq 15057 \\
                  & x_1 \leq 62440 \\
                  & x_2 \leq 75500 \\
                  & x_1, x_2 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_1$: units of 27in 4K Gaming Monitor fulfilled
- $x_2$: units of 27in FHD Monitor fulfilled