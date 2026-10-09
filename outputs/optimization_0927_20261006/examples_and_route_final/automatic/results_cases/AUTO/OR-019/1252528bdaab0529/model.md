Let $I$ be the set of products classified as ‘27in’:
\[
I = \{\text{27in 4K Gaming Monitor},\ \text{27in FHD Monitor}\}
\]

Let $x_i$ be the number of units of product $i \in I$ to fulfill.

Parameters (from the data):

\[
\begin{array}{l|ccc}
\text{Product Name} & \text{Revenue } r_i & \text{Demand } d_i & \text{Initial Inventory } s_i \\
\hline
\text{27in 4K Gaming Monitor} & 389.99 & 12474 & 62440 \\
\text{27in FHD Monitor} & 149.99 & 15057 & 75500 \\
\end{array}
\]

Objective:
\[
\max \quad 389.99\, x_{\text{27in 4K Gaming Monitor}} + 149.99\, x_{\text{27in FHD Monitor}}
\]

Subject to, for each $i \in I$:
\[
\begin{align*}
0 \leq x_i \leq \min\{d_i,\, s_i\} \\
x_i \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Explicitly:
\[
\begin{align*}
0 \leq x_{\text{27in 4K Gaming Monitor}} \leq 12474 \\
0 \leq x_{\text{27in FHD Monitor}} \leq 15057 \\
x_{\text{27in 4K Gaming Monitor}},\ x_{\text{27in FHD Monitor}} \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $i \in I$)
- $r_i$ = revenue per unit of product $i$
- $d_i$ = demand for product $i$
- $s_i$ = initial inventory for product $i$

All coefficients and bounds are taken directly from the retrieved data, preserving source order and identifiers.