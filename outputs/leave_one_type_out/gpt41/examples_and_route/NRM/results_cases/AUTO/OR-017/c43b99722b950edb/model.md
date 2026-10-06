Let $I$ be the set of SKUs classified as ‘ZZ’:
$$
I = \{\text{ZZ2AO},\ \text{ZZDW7},\ \text{ZZM1A},\ \text{ZZNC5},\ \text{ZZX6K}\}
$$

Let $x_i$ be the number of units of SKU $i$ to fulfill, for each $i \in I$.

Parameters (from the data):

\[
\begin{array}{lcccc}
\text{SKU} & \text{Revenue}_i & \text{InitialInventory}_i & \text{Demand}_i \\
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
\max \sum_{i \in I} \text{Revenue}_i \cdot x_i
\]

Subject to, for each $i \in I$:
\[
\begin{align*}
x_i &\leq \text{InitialInventory}_i \\
x_i &\leq \text{Demand}_i \\
x_i &\geq 0,\quad x_i \in \mathbb{Z}
\end{align*}
\]

Explicitly, the model is:

\[
\begin{align*}
\max\quad & 24.38\, x_{\text{ZZ2AO}} + 30.12\, x_{\text{ZZDW7}} + 19.52\, x_{\text{ZZM1A}} + 10.79\, x_{\text{ZZNC5}} + 111.81\, x_{\text{ZZX6K}} \\
\text{s.t.}\quad
& x_{\text{ZZ2AO}} \leq 10.0 \\
& x_{\text{ZZ2AO}} \leq 2 \\
& x_{\text{ZZDW7}} \leq 20.0 \\
& x_{\text{ZZDW7}} \leq 4 \\
& x_{\text{ZZM1A}} \leq 530.0 \\
& x_{\text{ZZM1A}} \leq 82 \\
& x_{\text{ZZNC5}} \leq 10.0 \\
& x_{\text{ZZNC5}} \leq 2 \\
& x_{\text{ZZX6K}} \leq 10.0 \\
& x_{\text{ZZX6K}} \leq 2 \\
& x_i \geq 0,\quad x_i \in \mathbb{Z},\quad \forall i \in I
\end{align*}
\]