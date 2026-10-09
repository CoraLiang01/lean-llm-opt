Let $I$ be the set of products classified as ‘ELE-S’, indexed by their Product_Reference.

Let $x_i$ = number of units of product $i$ to fulfill, for each $i \in I$.

Parameters (from the data):

\[
\begin{array}{llll}
\text{Product\_Reference} & \text{Revenue} & \text{Demand} & \text{Initial Inventory} \\
\hline
\text{ELE-SMA-10000463} & 4.0 & 295 & 2000.0 \\
\text{ELE-SMA-10000487} & 14.0 & 1002 & 7000.0 \\
\text{ELE-SMA-10003333} & 14.0 & 958 & 7000.0 \\
\text{ELE-SMA-10009012} & 4.0 & 777 & 6000.0 \\
\text{ELE-SMA-10009999} & 4.0 & 271 & 2000.0 \\
\text{ELE-SMA-10011234} & 4.0 & 244 & 2000.0 \\
\text{ELE-SMA-10027456} & 14.0 & 990 & 7000.0 \\
\text{ELE-SMA-10028567} & 14.0 & 1000 & 7000.0 \\
\end{array}
\]

Model:

Objective:
\[
\max \sum_{i \in I} \text{Revenue}_i \cdot x_i
\]

Subject to, for each $i \in I$:
\[
0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}
\]
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

Explicitly, for each product:

\[
\begin{align*}
0 \leq x_{\text{ELE-SMA-10000463}} &\leq 295 \\
0 \leq x_{\text{ELE-SMA-10000487}} &\leq 1002 \\
0 \leq x_{\text{ELE-SMA-10003333}} &\leq 958 \\
0 \leq x_{\text{ELE-SMA-10009012}} &\leq 777 \\
0 \leq x_{\text{ELE-SMA-10009999}} &\leq 271 \\
0 \leq x_{\text{ELE-SMA-10011234}} &\leq 244 \\
0 \leq x_{\text{ELE-SMA-10027456}} &\leq 990 \\
0 \leq x_{\text{ELE-SMA-10028567}} &\leq 1000 \\
\end{align*}
\]

and all $x_i$ are nonnegative integers.