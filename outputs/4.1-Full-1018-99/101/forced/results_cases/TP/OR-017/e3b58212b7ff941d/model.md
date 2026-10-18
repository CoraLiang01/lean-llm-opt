Let $I$ be the set of ‘ZZ’ products:
$$
I = \{\text{ZZ2AO},\ \text{ZZDW7},\ \text{ZZM1A},\ \text{ZZNC5},\ \text{ZZX6K}\}
$$

Define decision variables:
$$
x_i \geq 0 \quad \text{(integer or continuous, as appropriate)} \\
\text{Number of units of product } i \text{ to fulfill, for each } i \in I
$$

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
\[
\max \sum_{i \in I} \text{Revenue}_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& 0 \leq x_i \leq \min(\text{Demand}_i,\, \text{Initial Inventory}_i) \qquad \forall i \in I \\
\end{align*}
\]

Explicitly, the constraints are:
\[
\begin{align*}
0 \leq x_{\text{ZZ2AO}} &\leq 2 \\
0 \leq x_{\text{ZZDW7}} &\leq 4 \\
0 \leq x_{\text{ZZM1A}} &\leq 82 \\
0 \leq x_{\text{ZZNC5}} &\leq 2 \\
0 \leq x_{\text{ZZX6K}} &\leq 2 \\
\end{align*}
\]

All $x_i$ are nonnegative and cannot exceed either the available inventory or the demand for each product.

Complete numerical formulation:

\[
\begin{align*}
\max\quad & 24.38\,x_{\text{ZZ2AO}} + 30.12\,x_{\text{ZZDW7}} + 19.52\,x_{\text{ZZM1A}} + 10.79\,x_{\text{ZZNC5}} + 111.81\,x_{\text{ZZX6K}} \\
\text{s.t.}\quad
& 0 \leq x_{\text{ZZ2AO}} \leq 2 \\
& 0 \leq x_{\text{ZZDW7}} \leq 4 \\
& 0 \leq x_{\text{ZZM1A}} \leq 82 \\
& 0 \leq x_{\text{ZZNC5}} \leq 2 \\
& 0 \leq x_{\text{ZZX6K}} \leq 2 \\
\end{align*}
\]
with $x_i \geq 0$ for all $i \in I$.

All coefficients and bounds are taken directly from the retrieved data.