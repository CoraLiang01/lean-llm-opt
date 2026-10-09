Let $I$ be the set of ‘27in’ products:
\[
I = \{\text{27in 4K Gaming Monitor},\ \text{27in FHD Monitor}\}
\]

Let $x_i$ be the number of units of product $i \in I$ to fulfill (decision variable, continuous and nonnegative).

Parameters (from source, in order):

\[
\begin{array}{l|c|c|c}
\text{Product Name} & \text{Revenue}_i & \text{Demand}_i & \text{Initial Inventory}_i \\
\hline
\text{27in 4K Gaming Monitor} & 389.99 & 12474 & 62440 \\
\text{27in FHD Monitor} & 149.99 & 15057 & 75500 \\
\end{array}
\]

Model:

Maximize total revenue:
\[
\max\ 389.99\,x_1 + 149.99\,x_2
\]

Subject to:
\[
\begin{align*}
0 \leq x_1 \leq \min(12474,\ 62440) = 12474 \\
0 \leq x_2 \leq \min(15057,\ 75500) = 15057 \\
\end{align*}
\]

Or, explicitly:
\[
\begin{align*}
0 \leq x_1 \leq 12474 \\
0 \leq x_2 \leq 15057 \\
\end{align*}
\]

Where:
\[
x_1 = \text{units fulfilled of 27in 4K Gaming Monitor} \\
x_2 = \text{units fulfilled of 27in FHD Monitor}
\]

All variables are continuous and nonnegative.