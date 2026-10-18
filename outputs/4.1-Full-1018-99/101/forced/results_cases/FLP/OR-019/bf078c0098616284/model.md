##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ (where $i$ is a ‘27in’ product) to fulfill.

##### Parameters

Let $I$ be the set of ‘27in’ products:
- $I = \{\text{27in 4K Gaming Monitor},\ \text{27in FHD Monitor}\}$

For each $i \in I$:
- $\text{Revenue}_i$: revenue per unit of product $i$
- $\text{Demand}_i$: demand for product $i$
- $\text{Inventory}_i$: initial inventory for product $i$

From the data:
- $\text{Revenue}_{\text{27in 4K Gaming Monitor}} = 389.99$
- $\text{Demand}_{\text{27in 4K Gaming Monitor}} = 12474$
- $\text{Inventory}_{\text{27in 4K Gaming Monitor}} = 62440$

- $\text{Revenue}_{\text{27in FHD Monitor}} = 149.99$
- $\text{Demand}_{\text{27in FHD Monitor}} = 15057$
- $\text{Inventory}_{\text{27in FHD Monitor}} = 75500$

##### Objective Function

\[
\max \sum_{i \in I} \text{Revenue}_i \cdot x_i
\]
That is,
\[
\max\ 389.99\, x_{\text{27in 4K Gaming Monitor}} + 149.99\, x_{\text{27in FHD Monitor}}
\]

##### Constraints

For each $i \in I$:
1. Demand fulfillment: $x_i \leq \text{Demand}_i$
2. Inventory limit:   $x_i \leq \text{Inventory}_i$
3. Nonnegativity and integrality: $x_i \in \mathbb{Z}_{\geq 0}$

Explicitly:
\[
\begin{align*}
0 \leq x_{\text{27in 4K Gaming Monitor}} &\leq \min(12474,\ 62440) = 12474 \\
0 \leq x_{\text{27in FHD Monitor}} &\leq \min(15057,\ 75500) = 15057 \\
x_{\text{27in 4K Gaming Monitor}},\ x_{\text{27in FHD Monitor}} &\in \mathbb{Z}_{\geq 0}
\end{align*}
\]

##### Complete Model

\[
\begin{align*}
\max\quad & 389.99\, x_{\text{27in 4K Gaming Monitor}} + 149.99\, x_{\text{27in FHD Monitor}} \\
\text{s.t.}\quad
& 0 \leq x_{\text{27in 4K Gaming Monitor}} \leq 12474 \\
& 0 \leq x_{\text{27in FHD Monitor}} \leq 15057 \\
& x_{\text{27in 4K Gaming Monitor}},\ x_{\text{27in FHD Monitor}} \in \mathbb{Z}_{\geq 0}
\end{align*}
\]