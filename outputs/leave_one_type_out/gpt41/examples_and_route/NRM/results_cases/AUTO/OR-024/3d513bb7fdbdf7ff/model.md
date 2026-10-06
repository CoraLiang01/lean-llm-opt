Let $x_i$ be the number of units of product $i$ (with ProductID as below) to fulfill.

Objective:
$$
\max \sum_{i} r_i x_i
$$
where $r_i$ is the Revenue for product $i$.

Subject to, for each product $i$:
\[
\begin{align*}
& x_i \leq \text{Initial Inventory}_i \\
& x_i \leq \text{Demand}_i \\
& x_i \geq 0,\quad x_i \in \mathbb{Z}
\end{align*}
\]

Numerical formulation with retrieved data:

Let
\[
\begin{array}{llll}
\text{ProductID} & r_i & \text{Initial Inventory}_i & \text{Demand}_i \\
\hline
\text{S700\_1138} & 70.67 & 9020 & 1219 \\
\text{S700\_1691} & 100.0 & 8370 & 1127 \\
\text{S700\_1938} & 70.15 & 8390 & 1129 \\
\text{S700\_2047} & 100.0 & 8680 & 1176 \\
\text{S700\_2466} & 100.0 & 9400 & 1301 \\
\text{S700\_2610} & 65.77 & 9900 & 1340 \\
\text{S700\_2824} & 100.0 & 9760 & 1357 \\
\text{S700\_2834} & 100.0 & 8610 & 1158 \\
\text{S700\_3167} & 74.4 & 9380 & 1287 \\
\text{S700\_3505} & 81.14 & 9170 & 1281 \\
\text{S700\_3962} & 100.0 & 8520 & 1135 \\
\text{S700\_4002} & 61.44 & 10290 & 1392 \\
\end{array}
\]

Variables:
\[
x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in \{\text{S700\_1138}, \text{S700\_1691}, \ldots, \text{S700\_4002}\}
\]

Objective:
\[
\max\ 
70.67\,x_{\text{S700\_1138}} + 100.0\,x_{\text{S700\_1691}} + 70.15\,x_{\text{S700\_1938}} + 100.0\,x_{\text{S700\_2047}} + 100.0\,x_{\text{S700\_2466}} + 65.77\,x_{\text{S700\_2610}} + 100.0\,x_{\text{S700\_2824}} + 100.0\,x_{\text{S700\_2834}} + 74.4\,x_{\text{S700\_3167}} + 81.14\,x_{\text{S700\_3505}} + 100.0\,x_{\text{S700\_3962}} + 61.44\,x_{\text{S700\_4002}}
\]

Subject to, for each $i$:
\[
\begin{align*}
x_{\text{S700\_1138}} &\leq 9020 \\
x_{\text{S700\_1138}} &\leq 1219 \\
x_{\text{S700\_1691}} &\leq 8370 \\
x_{\text{S700\_1691}} &\leq 1127 \\
x_{\text{S700\_1938}} &\leq 8390 \\
x_{\text{S700\_1938}} &\leq 1129 \\
x_{\text{S700\_2047}} &\leq 8680 \\
x_{\text{S700\_2047}} &\leq 1176 \\
x_{\text{S700\_2466}} &\leq 9400 \\
x_{\text{S700\_2466}} &\leq 1301 \\
x_{\text{S700\_2610}} &\leq 9900 \\
x_{\text{S700\_2610}} &\leq 1340 \\
x_{\text{S700\_2824}} &\leq 9760 \\
x_{\text{S700\_2824}} &\leq 1357 \\
x_{\text{S700\_2834}} &\leq 8610 \\
x_{\text{S700\_2834}} &\leq 1158 \\
x_{\text{S700\_3167}} &\leq 9380 \\
x_{\text{S700\_3167}} &\leq 1287 \\
x_{\text{S700\_3505}} &\leq 9170 \\
x_{\text{S700\_3505}} &\leq 1281 \\
x_{\text{S700\_3962}} &\leq 8520 \\
x_{\text{S700\_3962}} &\leq 1135 \\
x_{\text{S700\_4002}} &\leq 10290 \\
x_{\text{S700\_4002}} &\leq 1392 \\
x_i &\geq 0,\quad x_i \in \mathbb{Z},\quad \forall i
\end{align*}
\]

Or, equivalently, for each $i$:
\[
0 \leq x_i \leq \min\{\text{Initial Inventory}_i,\, \text{Demand}_i\},\quad x_i \in \mathbb{Z}
\]