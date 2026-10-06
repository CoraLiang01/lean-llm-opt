Let $x_i$ be the number of units of product $i$ (with Product_Reference as below) to fulfill.

Objective:
$$
\max \; 4.0\,x_{\text{ELE-SMA-10000463}} + 14.0\,x_{\text{ELE-SMA-10000487}} + 14.0\,x_{\text{ELE-SMA-10003333}} + 4.0\,x_{\text{ELE-SMA-10009012}} + 4.0\,x_{\text{ELE-SMA-10009999}} + 4.0\,x_{\text{ELE-SMA-10011234}} + 14.0\,x_{\text{ELE-SMA-10027456}} + 14.0\,x_{\text{ELE-SMA-10028567}}
$$

Subject to, for each product $i$:

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

Inventory constraints:
\[
\begin{align*}
x_{\text{ELE-SMA-10000463}} &\leq 2000 \\
x_{\text{ELE-SMA-10000487}} &\leq 7000 \\
x_{\text{ELE-SMA-10003333}} &\leq 7000 \\
x_{\text{ELE-SMA-10009012}} &\leq 6000 \\
x_{\text{ELE-SMA-10009999}} &\leq 2000 \\
x_{\text{ELE-SMA-10011234}} &\leq 2000 \\
x_{\text{ELE-SMA-10027456}} &\leq 7000 \\
x_{\text{ELE-SMA-10028567}} &\leq 7000 \\
\end{align*}
\]

Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \text{for all products } i
\]

Where:
- $x_i$ = number of units of product $i$ to fulfill
- Product references and parameters:

| Product_Reference      | Revenue | Demand | Initial Inventory |
|-----------------------|---------|--------|------------------|
| ELE-SMA-10000463      | 4.0     | 295    | 2000.0           |
| ELE-SMA-10000487      | 14.0    | 1002   | 7000.0           |
| ELE-SMA-10003333      | 14.0    | 958    | 7000.0           |
| ELE-SMA-10009012      | 4.0     | 777    | 6000.0           |
| ELE-SMA-10009999      | 4.0     | 271    | 2000.0           |
| ELE-SMA-10011234      | 4.0     | 244    | 2000.0           |
| ELE-SMA-10027456      | 14.0    | 990    | 7000.0           |
| ELE-SMA-10028567      | 14.0    | 1000   | 7000.0           |

For each $i$, $x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}$.