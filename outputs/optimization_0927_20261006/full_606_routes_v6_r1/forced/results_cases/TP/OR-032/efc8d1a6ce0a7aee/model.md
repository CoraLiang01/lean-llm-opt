Let $I$ be the set of all products with names starting with "Books_", i.e.,
$$
I = \{\text{Books\_15.15},\ \text{Books\_30.3},\ \text{Books\_45.45},\ \text{Books\_60.6},\ \text{Books\_75.75}\}
$$

Let $x_i$ be the number of units of product $i \in I$ to fulfill (continuous, $x_i \geq 0$).

Parameters (from the data):

\[
\begin{array}{l|cccccc}
\text{Product\_Name} & \text{Revenue}_i & \text{Demand}_i & \text{Initial Inventory}_i \\
\hline
\text{Books\_15.15} & 15.15 & 1980 & 9920.0 \\
\text{Books\_30.3} & 30.3 & 3024 & 20160.0 \\
\text{Books\_45.45} & 45.45 & 4536 & 30000.0 \\
\text{Books\_60.6} & 60.6 & 5601 & 38360.0 \\
\text{Books\_75.75} & 75.75 & 7567 & 51450.0 \\
\end{array}
\]

Model:

Maximize total revenue:
\[
\max \quad 15.15\,x_{\text{Books\_15.15}} + 30.3\,x_{\text{Books\_30.3}} + 45.45\,x_{\text{Books\_45.45}} + 60.6\,x_{\text{Books\_60.6}} + 75.75\,x_{\text{Books\_75.75}}
\]

Subject to, for each $i \in I$:
\[
0 \leq x_i \leq \min\{\text{Initial Inventory}_i,\, \text{Demand}_i\}
\]
That is,
\[
\begin{align*}
0 \leq x_{\text{Books\_15.15}} &\leq 1980 \\
0 \leq x_{\text{Books\_30.3}} &\leq 3024 \\
0 \leq x_{\text{Books\_45.45}} &\leq 4536 \\
0 \leq x_{\text{Books\_60.6}} &\leq 5601 \\
0 \leq x_{\text{Books\_75.75}} &\leq 7567 \\
\end{align*}
\]

All $x_i$ are continuous and nonnegative.