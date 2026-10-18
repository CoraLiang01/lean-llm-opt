Let $I$ be the set of FAUX products, indexed in the source order as follows:

\[
\begin{array}{lllr}
i & \text{Product Name} & \text{Revenue}_i & \text{Initial Inventory}_i & \text{Demand}_i \\
1 & \text{FAUX FUR JEWEL SWEATER} & 35.9 & 20970 & 3025 \\
2 & \text{FAUX LEATHER BOMBER JACKET} & 69.9 & 71970 & 9585 \\
3 & \text{FAUX LEATHER BOXY FIT JACKET} & 99.9 & 32730 & 4486 \\
4 & \text{FAUX LEATHER JACKET} & 99.9 & 71130 & 10322 \\
5 & \text{FAUX LEATHER OVERSIZED JACKET LIMITED EDITION} & 159.0 & 34910 & 4868 \\
6 & \text{FAUX LEATHER PUFFER JACKET} & 69.99 & 64010 & 8482 \\
7 & \text{FAUX SHEARLING LINED SUEDE BOOTS} & 99.9 & 20760 & 2607 \\
8 & \text{FAUX SHEARLING PLAID JACKET} & 89.9 & 12490 & 1784 \\
9 & \text{FAUX SUEDE BOMBER JACKET} & 69.9 & 50300 & 6626 \\
10 & \text{FAUX SUEDE JACKET} & 89.9 & 24570 & 3256 \\
11 & \text{FAUX SUEDE OVERSHIRT} & 69.9 & 24430 & 2955 \\
12 & \text{FAUX SUEDE PATCH JACKET} & 89.9 & 7070 & 910 \\
\end{array}
\]

Decision variables:

For each $i\in I$, let $x_i\geq 0$ be the (continuous) number of units of product $i$ fulfilled.

Objective:

\[
\max \sum_{i\in I} \text{Revenue}_i \cdot x_i
\]

Constraints:

For each $i\in I$:
\[
0 \leq x_i \leq \min\{\text{Initial Inventory}_i,\, \text{Demand}_i\}
\]

Numerically, for each product $i$:

\[
\begin{align*}
0 \leq x_1 &\leq 3025 \\
0 \leq x_2 &\leq 9585 \\
0 \leq x_3 &\leq 4486 \\
0 \leq x_4 &\leq 10322 \\
0 \leq x_5 &\leq 4868 \\
0 \leq x_6 &\leq 8482 \\
0 \leq x_7 &\leq 2607 \\
0 \leq x_8 &\leq 1784 \\
0 \leq x_9 &\leq 6626 \\
0 \leq x_{10} &\leq 3256 \\
0 \leq x_{11} &\leq 2955 \\
0 \leq x_{12} &\leq 910 \\
\end{align*}
\]

All variables $x_i$ are continuous and nonnegative.

Complete model:

\[
\begin{align*}
\max\quad & 35.9\,x_1 + 69.9\,x_2 + 99.9\,x_3 + 99.9\,x_4 + 159.0\,x_5 + 69.99\,x_6 \\
&\quad + 99.9\,x_7 + 89.9\,x_8 + 69.9\,x_9 + 89.9\,x_{10} + 69.9\,x_{11} + 89.9\,x_{12} \\
\text{s.t.}\quad
& 0 \leq x_1 \leq 3025 \\
& 0 \leq x_2 \leq 9585 \\
& 0 \leq x_3 \leq 4486 \\
& 0 \leq x_4 \leq 10322 \\
& 0 \leq x_5 \leq 4868 \\
& 0 \leq x_6 \leq 8482 \\
& 0 \leq x_7 \leq 2607 \\
& 0 \leq x_8 \leq 1784 \\
& 0 \leq x_9 \leq 6626 \\
& 0 \leq x_{10} \leq 3256 \\
& 0 \leq x_{11} \leq 2955 \\
& 0 \leq x_{12} \leq 910 \\
& x_i \geq 0,\quad \forall i\in I
\end{align*}
\]