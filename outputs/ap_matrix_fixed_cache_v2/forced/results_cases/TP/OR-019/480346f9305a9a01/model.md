Let $I$ be the set of products classified as ‘27in’:
\[
I = \{\text{27in 4K Gaming Monitor},\ \text{27in FHD Monitor}\}
\]

Decision variables:
\[
x_i \geq 0 \quad \text{(continuous)},\quad \forall i \in I
\]
where $x_i$ is the number of units of product $i$ to fulfill.

Parameters (in source order):

\[
\begin{array}{l|c|c|c}
\text{Product Name} & \text{Revenue}_i & \text{Demand}_i & \text{Initial Inventory}_i \\
\hline
\text{27in 4K Gaming Monitor} & 389.99 & 12474 & 62440 \\
\text{27in FHD Monitor} & 149.99 & 15057 & 75500 \\
\end{array}
\]

Objective:
\[
\max\ 389.99\,x_{\text{27in 4K Gaming Monitor}} + 149.99\,x_{\text{27in FHD Monitor}}
\]

Subject to:
\[
\begin{align*}
0 \leq x_{\text{27in 4K Gaming Monitor}} &\leq \min\{12474,\ 62440\} = 12474 \\
0 \leq x_{\text{27in FHD Monitor}} &\leq \min\{15057,\ 75500\} = 15057 \\
\end{align*}
\]

Or, equivalently, for each $i \in I$:
\[
0 \leq x_i \leq \min\{\text{Demand}_i,\ \text{Initial Inventory}_i\}
\]

All parameters and identifiers are preserved in source order.