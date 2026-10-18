##### Sets and Indices

Let $I$ be the set of products classified under ‘27in’:
$$
I = \{\text{27in 4K Gaming Monitor},\ \text{27in FHD Monitor}\}
$$

##### Parameters

For each $i \in I$:
- $r_i$: Revenue per unit of product $i$
- $d_i$: Demand for product $i$
- $s_i$: Initial inventory of product $i$

From the data:
- $r_{\text{27in 4K Gaming Monitor}} = 261.2933$
- $r_{\text{27in FHD Monitor}} = 52.4965$
- $d_{\text{27in 4K Gaming Monitor}} = 12474$
- $d_{\text{27in FHD Monitor}} = 15057$
- $s_{\text{27in 4K Gaming Monitor}} = 62440$
- $s_{\text{27in FHD Monitor}} = 75500$

##### Decision Variables

For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$)

##### Mathematical Model

Maximize total revenue:
$$
\max\ 261.2933\,x_{\text{27in 4K Gaming Monitor}} + 52.4965\,x_{\text{27in FHD Monitor}}
$$

Subject to, for each $i \in I$:
- Demand fulfillment cannot exceed demand:
  $$
  x_i \leq d_i
  $$
- Fulfillment cannot exceed initial inventory:
  $$
  x_i \leq s_i
  $$
- Nonnegativity and integrality:
  $$
  x_i \geq 0,\quad x_i \in \mathbb{Z}
  $$

Explicitly, the model is:

Maximize
$$
261.2933\,x_{\text{27in 4K Gaming Monitor}} + 52.4965\,x_{\text{27in FHD Monitor}}
$$

Subject to
\begin{align*}
x_{\text{27in 4K Gaming Monitor}} &\leq 12474 \\
x_{\text{27in 4K Gaming Monitor}} &\leq 62440 \\
x_{\text{27in FHD Monitor}} &\leq 15057 \\
x_{\text{27in FHD Monitor}} &\leq 75500 \\
x_{\text{27in 4K Gaming Monitor}},\ x_{\text{27in FHD Monitor}} &\geq 0,\ \text{integer}
\end{align*}