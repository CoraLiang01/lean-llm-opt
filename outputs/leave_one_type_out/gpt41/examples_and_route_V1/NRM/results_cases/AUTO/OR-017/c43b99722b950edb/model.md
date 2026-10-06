Let $I$ be the set of products classified as ‘ZZ’:
$$
I = \{\text{ZZ2AO},\ \text{ZZDW7},\ \text{ZZM1A},\ \text{ZZNC5},\ \text{ZZX6K}\}
$$

Let $x_i$ be the number of units of product $i \in I$ to fulfill.

Parameters (from the data):

\[
\begin{array}{l|c|c|c}
\text{SKU} & \text{Revenue}_i & \text{Initial Inventory}_i & \text{Demand}_i \\
\hline
\text{ZZ2AO} & 24.38 & 10.0 & 2 \\
\text{ZZDW7} & 30.12 & 20.0 & 4 \\
\text{ZZM1A} & 19.52 & 530.0 & 82 \\
\text{ZZNC5} & 10.79 & 10.0 & 2 \\
\text{ZZX6K} & 111.81 & 10.0 & 2 \\
\end{array}
\]

Objective:
\[
\max \quad 24.38\,x_{\text{ZZ2AO}} + 30.12\,x_{\text{ZZDW7}} + 19.52\,x_{\text{ZZM1A}} + 10.79\,x_{\text{ZZNC5}} + 111.81\,x_{\text{ZZX6K}}
\]

Subject to, for each $i \in I$:
\[
\begin{align*}
x_i &\leq \text{Initial Inventory}_i \\
x_i &\leq \text{Demand}_i \\
x_i &\geq 0,\quad x_i \in \mathbb{Z}
\end{align*}
\]

Explicitly, the constraints are:
\[
\begin{align*}
0 \leq x_{\text{ZZ2AO}} &\leq \min\{10,\ 2\} = 2 \\
0 \leq x_{\text{ZZDW7}} &\leq \min\{20,\ 4\} = 4 \\
0 \leq x_{\text{ZZM1A}} &\leq \min\{530,\ 82\} = 82 \\
0 \leq x_{\text{ZZNC5}} &\leq \min\{10,\ 2\} = 2 \\
0 \leq x_{\text{ZZX6K}} &\leq \min\{10,\ 2\} = 2 \\
\end{align*}
\]

All $x_i$ are nonnegative integers.