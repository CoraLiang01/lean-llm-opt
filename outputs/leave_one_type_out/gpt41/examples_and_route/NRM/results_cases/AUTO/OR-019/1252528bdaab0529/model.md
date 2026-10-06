Let $x_i$ be the number of units of 27in product $i$ to fulfill, for each product $i$ in:

- 27in 4K Gaming Monitor
- 27in FHD Monitor

Parameters:

- $r_i$: Revenue per unit of product $i$
- $d_i$: Demand for product $i$
- $s_i$: Initial Inventory of product $i$

Data:

\[
\begin{array}{lccc}
\text{Product Name} & r_i & d_i & s_i \\
\hline
\text{27in 4K Gaming Monitor} & 389.99 & 12474 & 62440 \\
\text{27in FHD Monitor} & 149.99 & 15057 & 75500 \\
\end{array}
\]

Decision variables:

\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i
\]

Objective:

\[
\max \; 389.99\, x_{\text{27in 4K Gaming Monitor}} + 149.99\, x_{\text{27in FHD Monitor}}
\]

Subject to:

\[
\begin{align*}
& x_{\text{27in 4K Gaming Monitor}} \leq 12474 \\
& x_{\text{27in FHD Monitor}} \leq 15057 \\
& x_{\text{27in 4K Gaming Monitor}} \leq 62440 \\
& x_{\text{27in FHD Monitor}} \leq 75500 \\
& x_{\text{27in 4K Gaming Monitor}},\; x_{\text{27in FHD Monitor}} \geq 0 \text{ and integer}
\end{align*}
\]

Where:

- The first two constraints ensure that the number of units fulfilled does not exceed demand.
- The next two constraints ensure that the number of units fulfilled does not exceed initial inventory.
- Variables are nonnegative integers.