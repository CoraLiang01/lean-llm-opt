Let $x_i$ denote the number of units of each ‘Baby’ product $i$ to be fulfilled.

Objective:
$$
\max \sum_{i} r_i x_i
$$

Subject to:
\[
\begin{align*}
& x_i \leq \text{Initial Inventory}_i, \quad \forall i \\
& x_i \leq \text{Demand}_i, \quad \forall i \\
& x_i \geq 0,\ x_i \in \mathbb{Z}, \quad \forall i
\end{align*}
\]

Where:

- For the retrieved data (in source order):

| Product Name         | Revenue ($r_i$) | Initial Inventory | Demand  |
|--------------------- |-----------------|------------------|---------|
| Baby Food_255.28     | 255.28          | 5,627,060        | 765,850 |

So, the explicit model is:

Let $x_1$ = units of "Baby Food_255.28" fulfilled.

Objective:
$$
\max\ 255.28\, x_1
$$

Subject to:
\[
\begin{align*}
& x_1 \leq 5,\!627,\!060 \\
& x_1 \leq 765,\!850 \\
& x_1 \geq 0,\ x_1 \in \mathbb{Z}
\end{align*}
\]