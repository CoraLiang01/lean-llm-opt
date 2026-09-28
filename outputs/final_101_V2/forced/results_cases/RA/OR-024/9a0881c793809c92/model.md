Let $x_i$ denote the number of units of product $i$ (with identifier as below) to fulfill.

Objective:
$$
\max \; 70.67\, x_{\text{S700\_1138}} + 100.0\, x_{\text{S700\_1691}} + 70.15\, x_{\text{S700\_1938}} + 100.0\, x_{\text{S700\_2047}} + 100.0\, x_{\text{S700\_2466}} + 65.77\, x_{\text{S700\_2610}} + 100.0\, x_{\text{S700\_2824}} + 100.0\, x_{\text{S700\_2834}} + 74.4\, x_{\text{S700\_3167}} + 81.14\, x_{\text{S700\_3505}} + 100.0\, x_{\text{S700\_3962}} + 61.44\, x_{\text{S700\_4002}}
$$

Subject to, for each product $i$:

- Demand constraint:
  $$
  x_i \leq \text{Demand}_i
  $$
- Inventory constraint:
  $$
  x_i \leq \text{Initial Inventory}_i
  $$
- Nonnegativity and integrality:
  $$
  x_i \in \mathbb{Z}_{\geq 0}
  $$

Where:

\[
\begin{array}{llll}
\text{Product Name} & \text{Revenue} & \text{Demand} & \text{Initial Inventory} \\
\hline
\text{S700\_1138} & 70.67 & 1219 & 9020 \\
\text{S700\_1691} & 100.0 & 1127 & 8370 \\
\text{S700\_1938} & 70.15 & 1129 & 8390 \\
\text{S700\_2047} & 100.0 & 1176 & 8680 \\
\text{S700\_2466} & 100.0 & 1301 & 9400 \\
\text{S700\_2610} & 65.77 & 1340 & 9900 \\
\text{S700\_2824} & 100.0 & 1357 & 9760 \\
\text{S700\_2834} & 100.0 & 1158 & 8610 \\
\text{S700\_3167} & 74.4 & 1287 & 9380 \\
\text{S700\_3505} & 81.14 & 1281 & 9170 \\
\text{S700\_3962} & 100.0 & 1135 & 8520 \\
\text{S700\_4002} & 61.44 & 1392 & 10290 \\
\end{array}
\]

Explicitly, for each $i$:

\[
\begin{align*}
0 \leq x_{\text{S700\_1138}} &\leq \min(1219, 9020) = 1219 \\
0 \leq x_{\text{S700\_1691}} &\leq \min(1127, 8370) = 1127 \\
0 \leq x_{\text{S700\_1938}} &\leq \min(1129, 8390) = 1129 \\
0 \leq x_{\text{S700\_2047}} &\leq \min(1176, 8680) = 1176 \\
0 \leq x_{\text{S700\_2466}} &\leq \min(1301, 9400) = 1301 \\
0 \leq x_{\text{S700\_2610}} &\leq \min(1340, 9900) = 1340 \\
0 \leq x_{\text{S700\_2824}} &\leq \min(1357, 9760) = 1357 \\
0 \leq x_{\text{S700\_2834}} &\leq \min(1158, 8610) = 1158 \\
0 \leq x_{\text{S700\_3167}} &\leq \min(1287, 9380) = 1287 \\
0 \leq x_{\text{S700\_3505}} &\leq \min(1281, 9170) = 1281 \\
0 \leq x_{\text{S700\_3962}} &\leq \min(1135, 8520) = 1135 \\
0 \leq x_{\text{S700\_4002}} &\leq \min(1392, 10290) = 1392 \\
\end{align*}
\]

All $x_i$ are nonnegative integers.