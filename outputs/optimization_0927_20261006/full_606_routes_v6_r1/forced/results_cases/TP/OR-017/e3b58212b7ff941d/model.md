Let $I$ be the set of SKUs classified as ‘ZZ’:
$$
I = \{\text{ZZ2AO},\ \text{ZZDW7},\ \text{ZZM1A},\ \text{ZZNC5},\ \text{ZZX6K}\}
$$

Decision variables:
$$
x_i \geq 0 \quad \text{(continuous)},\quad \forall i \in I
$$
where $x_i$ = number of units of SKU $i$ to fulfill.

Parameters (from retrieved data):

\[
\begin{array}{l|c|c|c}
\text{SKU} & \text{Revenue}_i & \text{Demand}_i & \text{Initial Inventory}_i \\
\hline
\text{ZZ2AO} & 24.38 & 2 & 10.0 \\
\text{ZZDW7} & 30.12 & 4 & 20.0 \\
\text{ZZM1A} & 19.52 & 82 & 530.0 \\
\text{ZZNC5} & 10.79 & 2 & 10.0 \\
\text{ZZX6K} & 111.81 & 2 & 10.0 \\
\end{array}
\]

Objective:
$$
\max \sum_{i \in I} \text{Revenue}_i \cdot x_i
$$

Constraints:
\[
\begin{align*}
& x_i \leq \text{Demand}_i, \quad \forall i \in I \\
& x_i \leq \text{Initial Inventory}_i, \quad \forall i \in I \\
& x_i \geq 0, \quad \forall i \in I
\end{align*}
\]

Explicitly, the model is:

\[
\max\ \ 24.38\,x_{\text{ZZ2AO}} + 30.12\,x_{\text{ZZDW7}} + 19.52\,x_{\text{ZZM1A}} + 10.79\,x_{\text{ZZNC5}} + 111.81\,x_{\text{ZZX6K}}
\]

subject to

\[
\begin{align*}
0 \leq x_{\text{ZZ2AO}} \leq \min\{2,\ 10.0\} = 2 \\
0 \leq x_{\text{ZZDW7}} \leq \min\{4,\ 20.0\} = 4 \\
0 \leq x_{\text{ZZM1A}} \leq \min\{82,\ 530.0\} = 82 \\
0 \leq x_{\text{ZZNC5}} \leq \min\{2,\ 10.0\} = 2 \\
0 \leq x_{\text{ZZX6K}} \leq \min\{2,\ 10.0\} = 2 \\
\end{align*}
\]

where all $x_i$ are continuous and nonnegative.