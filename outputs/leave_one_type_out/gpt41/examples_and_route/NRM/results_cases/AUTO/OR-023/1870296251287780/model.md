Let $x_i$ denote the number of units of product $i$ (with Product_Reference as below) to fulfill.

Objective:
$$
\max \; 4.0\, x_{\text{ELE-SMA-10000463}} + 14.0\, x_{\text{ELE-SMA-10000487}} + 14.0\, x_{\text{ELE-SMA-10003333}} + 4.0\, x_{\text{ELE-SMA-10009012}} + 4.0\, x_{\text{ELE-SMA-10009999}} + 4.0\, x_{\text{ELE-SMA-10011234}} + 14.0\, x_{\text{ELE-SMA-10027456}} + 14.0\, x_{\text{ELE-SMA-10028567}}
$$

Subject to, for each product $i$:

Inventory constraints:
\[
\begin{align*}
x_{\text{ELE-SMA-10000463}} &\leq 2000.0 \\
x_{\text{ELE-SMA-10000487}} &\leq 7000.0 \\
x_{\text{ELE-SMA-10003333}} &\leq 7000.0 \\
x_{\text{ELE-SMA-10009012}} &\leq 6000.0 \\
x_{\text{ELE-SMA-10009999}} &\leq 2000.0 \\
x_{\text{ELE-SMA-10011234}} &\leq 2000.0 \\
x_{\text{ELE-SMA-10027456}} &\leq 7000.0 \\
x_{\text{ELE-SMA-10028567}} &\leq 7000.0 \\
\end{align*}
\]

Demand constraints:
\[
\begin{align*}
x_{\text{ELE-SMA-10000463}} &\leq 295 \\
x_{\text{ELE-SMA-10000487}} &\leq 1002 \\
x_{\text{ELE-SMA-10003333}} &\leq 958 \\
x_{\text{ELE-SMA-10009012}} &\leq 777 \\
x_{\text{ELE-SMA-10009999}} &\leq 271 \\
x_{\text{ELE-SMA-10011234}} &\leq 244 \\
x_{\text{ELE-SMA-10027456}} &\leq 990 \\
x_{\text{ELE-SMA-10028567}} &\leq 1000 \\
\end{align*}
\]

Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all products } i
\]

Where:
- $x_i$ = number of units of product $i$ to fulfill
- Product references and coefficients as given above

For each product $i$, the actual fulfillment cannot exceed either the initial inventory or the demand:
\[
x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}
\]