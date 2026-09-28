Let $I$ be the set of SKUs classified under ‘ZZ’:
$$
I = \{\text{ZZ2AO},\ \text{ZZDW7},\ \text{ZZM1A},\ \text{ZZNC5},\ \text{ZZX6K}\}
$$

Let $x_i$ be the number of units of SKU $i$ to fulfill, for each $i \in I$.

Parameters (from the data):

\[
\begin{array}{l|ccc}
\text{SKU} & \text{Revenue}_i & \text{Demand}_i & \text{InitialInventory}_i \\
\hline
\text{ZZ2AO} & 24.38 & 2 & 10.0 \\
\text{ZZDW7} & 30.12 & 4 & 20.0 \\
\text{ZZM1A} & 19.52 & 82 & 530.0 \\
\text{ZZNC5} & 10.79 & 2 & 10.0 \\
\text{ZZX6K} & 111.81 & 2 & 10.0 \\
\end{array}
\]

Objective:
\[
\max \quad 24.38\,x_{\text{ZZ2AO}} + 30.12\,x_{\text{ZZDW7}} + 19.52\,x_{\text{ZZM1A}} + 10.79\,x_{\text{ZZNC5}} + 111.81\,x_{\text{ZZX6K}}
\]

Subject to, for each $i \in I$:
\[
\begin{align*}
x_i &\leq \text{Demand}_i \\
x_i &\leq \text{InitialInventory}_i \\
x_i &\geq 0 \\
x_i &\in \mathbb{Z}
\end{align*}
\]

Explicitly, the constraints are:
\[
\begin{align*}
0 \leq x_{\text{ZZ2AO}} &\leq \min\{2,\ 10.0\} \\
0 \leq x_{\text{ZZDW7}} &\leq \min\{4,\ 20.0\} \\
0 \leq x_{\text{ZZM1A}} &\leq \min\{82,\ 530.0\} \\
0 \leq x_{\text{ZZNC5}} &\leq \min\{2,\ 10.0\} \\
0 \leq x_{\text{ZZX6K}} &\leq \min\{2,\ 10.0\} \\
x_i &\in \mathbb{Z},\ \forall i \in I
\end{align*}
\]

Where $x_i$ is the number of units of SKU $i$ to fulfill, for each $i \in I$.