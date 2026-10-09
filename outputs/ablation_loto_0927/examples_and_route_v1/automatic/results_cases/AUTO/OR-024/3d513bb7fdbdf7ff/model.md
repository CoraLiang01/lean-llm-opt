Let $I$ be the set of products with IDs starting with ‘S700_’:

\[
I = \{
\text{S700\_1138},
\text{S700\_1691},
\text{S700\_1938},
\text{S700\_2047},
\text{S700\_2466},
\text{S700\_2610},
\text{S700\_2824},
\text{S700\_2834},
\text{S700\_3167},
\text{S700\_3505},
\text{S700\_3962},
\text{S700\_4002}
\}
\]

Let $x_i$ be the number of units of product $i \in I$ to fulfill.

Parameters for each $i \in I$:

- $r_i$: Revenue per unit
- $d_i$: Demand
- $s_i$: Initial Inventory

Data:

\[
\begin{array}{lccc}
\text{Product Name} & r_i & d_i & s_i \\
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

Model:

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Subject to, for all $i \in I$:
\[
0 \leq x_i \leq \min\{d_i, s_i\}
\]
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

Explicitly, for each product:

\[
\begin{align*}
0 \leq x_{\text{S700\_1138}} &\leq 1219 \\
0 \leq x_{\text{S700\_1691}} &\leq 1127 \\
0 \leq x_{\text{S700\_1938}} &\leq 1129 \\
0 \leq x_{\text{S700\_2047}} &\leq 1176 \\
0 \leq x_{\text{S700\_2466}} &\leq 1301 \\
0 \leq x_{\text{S700\_2610}} &\leq 1340 \\
0 \leq x_{\text{S700\_2824}} &\leq 1357 \\
0 \leq x_{\text{S700\_2834}} &\leq 1158 \\
0 \leq x_{\text{S700\_3167}} &\leq 1287 \\
0 \leq x_{\text{S700\_3505}} &\leq 1281 \\
0 \leq x_{\text{S700\_3962}} &\leq 1135 \\
0 \leq x_{\text{S700\_4002}} &\leq 1392 \\
\end{align*}
\]

where all $x_i$ are nonnegative integers.