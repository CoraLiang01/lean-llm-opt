Let $I$ be the set of products classified as ‘27in’:
\[
I = \{\text{27in 4K Gaming Monitor},\ \text{27in FHD Monitor}\}
\]

Let $r_i$ be the Revenue for product $i \in I$.

Let $d_i$ be the Demand for product $i \in I$.

Let $s_i$ be the Initial Inventory for product $i \in I$.

Let $x_i$ be the number of units of product $i$ to fulfill.

The model is:

\[
\begin{align*}
\max\quad & 389.99\, x_{\text{27in 4K Gaming Monitor}} + 149.99\, x_{\text{27in FHD Monitor}} \\[2ex]
\text{s.t.}\quad
& 0 \leq x_{\text{27in 4K Gaming Monitor}} \leq \min\{12474,\ 62440\} \\[1ex]
& 0 \leq x_{\text{27in FHD Monitor}} \leq \min\{15057,\ 75500\} \\[2ex]
& x_{\text{27in 4K Gaming Monitor}},\ x_{\text{27in FHD Monitor}} \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Or, numerically:

\[
\begin{align*}
\max\quad & 389.99\, x_1 + 149.99\, x_2 \\[2ex]
\text{s.t.}\quad
& 0 \leq x_1 \leq 12474 \\
& 0 \leq x_2 \leq 15057 \\
& x_1,\, x_2 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_1$ = units of 27in 4K Gaming Monitor to fulfill
- $x_2$ = units of 27in FHD Monitor to fulfill