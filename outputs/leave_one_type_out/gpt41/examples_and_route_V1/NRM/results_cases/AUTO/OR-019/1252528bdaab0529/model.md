Let $x_i$ be the number of units of product $i$ (where $i$ indexes the ‘27in’ products) that the company plans to fulfill.

Products:
- 27in 4K Gaming Monitor
- 27in FHD Monitor

Parameters:
- $r_i$: Revenue per unit of product $i$
- $s_i$: Initial Inventory of product $i$
- $d_i$: Demand for product $i$

Data:

| Product                     | $r_i$   | $s_i$  | $d_i$  |
|-----------------------------|---------|--------|--------|
| 27in 4K Gaming Monitor      | 389.99  | 62440  | 12474  |
| 27in FHD Monitor            | 149.99  | 75500  | 15057  |

Decision variables:
- $x_i \in \mathbb{Z}_{\geq 0}$, for each product $i$

Objective:
$$
\max \; 389.99\, x_{\text{4K}} + 149.99\, x_{\text{FHD}}
$$

Subject to:
\[
\begin{align*}
& x_{\text{4K}} \leq 62440 \\
& x_{\text{4K}} \leq 12474 \\
& x_{\text{FHD}} \leq 75500 \\
& x_{\text{FHD}} \leq 15057 \\
& x_{\text{4K}},\ x_{\text{FHD}} \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_{\text{4K}}$ = number of 27in 4K Gaming Monitors fulfilled
- $x_{\text{FHD}}$ = number of 27in FHD Monitors fulfilled

Each $x_i$ is bounded above by both its initial inventory and its demand. The objective is to maximize total revenue from fulfilling these products.