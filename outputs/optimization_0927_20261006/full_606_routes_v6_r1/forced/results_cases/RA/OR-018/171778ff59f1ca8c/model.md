Let $I$ be the set of Baby products (here, $I = \{\text{Baby Food\_255.28}\}$).

Let $x_i$ be the number of units of Baby product $i$ to fulfill, for $i \in I$.

Parameters:
- $r_i$: Revenue per unit of product $i$
- $d_i$: Demand for product $i$
- $s_i$: Initial Inventory of product $i$

Given data:
- For $\text{Baby Food\_255.28}$: $r_i = 255.28$, $d_i = 3,\!066,\!513$, $s_i = 22,\!749,\!210$

The model is:

$$
\begin{align*}
\max \quad & 255.28\, x_{\text{Baby Food\_255.28}} \\
\text{s.t.} \quad
& 0 \leq x_{\text{Baby Food\_255.28}} \leq 3,\!066,\!513 \\
& x_{\text{Baby Food\_255.28}} \leq 22,\!749,\!210 \\
& x_{\text{Baby Food\_255.28}} \in \mathbb{Z}_{\geq 0}
\end{align*}
$$

Or, more generally, for all $i \in I$:
$$
\begin{align*}
\max \quad & \sum_{i \in I} r_i x_i \\
\text{s.t.} \quad
& 0 \leq x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
$$

Where, for this instance, $I = \{\text{Baby Food\_255.28}\}$, $r_i = 255.28$, $d_i = 3,\!066,\!513$, $s_i = 22,\!749,\!210$.