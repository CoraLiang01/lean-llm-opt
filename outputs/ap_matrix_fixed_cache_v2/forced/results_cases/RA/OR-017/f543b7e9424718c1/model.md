Let $I$ be the set of SKUs classified as ‘ZZ’:
$$
I = \{\text{ZZ2AO},\ \text{ZZDW7},\ \text{ZZM1A},\ \text{ZZNC5},\ \text{ZZX6K}\}
$$

Let $x_i$ be the number of units of SKU $i$ to fulfill, for each $i \in I$.

Parameters (from the data):

\[
\begin{array}{l|c|c|c}
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
\max \sum_{i \in I} \text{Revenue}_i \cdot x_i
\]

Subject to, for each $i \in I$:
\[
\begin{align*}
0 \leq x_i &\leq \min\{\text{Demand}_i,\ \text{InitialInventory}_i\} \\
x_i &\in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Explicitly, for each SKU:

\[
\begin{align*}
0 \leq x_{\text{ZZ2AO}} &\leq 2 \\
0 \leq x_{\text{ZZDW7}} &\leq 4 \\
0 \leq x_{\text{ZZM1A}} &\leq 82 \\
0 \leq x_{\text{ZZNC5}} &\leq 2 \\
0 \leq x_{\text{ZZX6K}} &\leq 2 \\
x_i &\in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\end{align*}
\]

Where:
- $x_i$ = number of units of SKU $i$ to fulfill (decision variable, integer, $x_i \geq 0$)
- $\text{Revenue}_i$ = revenue per unit of SKU $i$
- $\text{Demand}_i$ = demand for SKU $i$
- $\text{InitialInventory}_i$ = initial inventory for SKU $i$

The model maximizes total revenue from fulfilling units of each ‘ZZ’ SKU, subject to not exceeding either demand or available inventory for each product.