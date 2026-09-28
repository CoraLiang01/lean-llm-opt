##### Decision Variables

$x_i \geq 0$: Number of units of product $i$ (with $i$ indexing all '27in' products) to fulfill.

##### Parameters

Let $I$ be the set of all '27in' products:

- $I = \{\text{27in 4K Gaming Monitor},\ \text{27in FHD Monitor}\}$

For each $i \in I$:

- $\text{Revenue}_i$: revenue per unit of product $i$
- $\text{Inventory}_i$: initial inventory of product $i$
- $\text{Demand}_i$: demand for product $i$

From the data:

| $i$                        | $\text{Revenue}_i$ | $\text{Inventory}_i$ | $\text{Demand}_i$ |
|----------------------------|--------------------|----------------------|-------------------|
| 27in 4K Gaming Monitor     | 261.2933           | 62440                | 12474             |
| 27in FHD Monitor           | 52.4965            | 75500                | 15057             |

##### Objective Function

\[
\max \sum_{i \in I} \text{Revenue}_i \cdot x_i
\]

##### Constraints

1. Inventory and demand limits:
   \[
   0 \leq x_i \leq \min\{\text{Inventory}_i,\, \text{Demand}_i\}, \quad \forall i \in I
   \]

2. $x_i$ are integer variables (if only whole units can be fulfilled), or continuous and nonnegative if partial fulfillment is allowed.

##### Complete Model

\[
\begin{align*}
\max\quad & 261.2933\, x_{\text{27in 4K Gaming Monitor}} + 52.4965\, x_{\text{27in FHD Monitor}} \\
\text{s.t.}\quad
& 0 \leq x_{\text{27in 4K Gaming Monitor}} \leq \min\{62440,\, 12474\} = 12474 \\
& 0 \leq x_{\text{27in FHD Monitor}} \leq \min\{75500,\, 15057\} = 15057 \\
& x_{\text{27in 4K Gaming Monitor}},\ x_{\text{27in FHD Monitor}} \geq 0 \\
& x_{\text{27in 4K Gaming Monitor}},\ x_{\text{27in FHD Monitor}} \in \mathbb{Z} \quad \text{(if integer units required)}
\end{align*}
\]

Where:
- $x_{\text{27in 4K Gaming Monitor}}$: units of 27in 4K Gaming Monitor fulfilled
- $x_{\text{27in FHD Monitor}}$: units of 27in FHD Monitor fulfilled

All parameters are as retrieved from the CSV data.