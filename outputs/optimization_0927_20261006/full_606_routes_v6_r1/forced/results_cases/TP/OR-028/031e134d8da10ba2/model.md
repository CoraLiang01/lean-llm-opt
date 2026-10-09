Let $I$ be the set of products, indexed in the source order as follows:

\[
I = \{\text{sku\_I27},\ \text{sku\_I499},\ \text{sku\_I719},\ \text{sku\_T18},\ \text{sku\_T29},\ \text{sku\_T39},\ \text{sku\_T499},\ \text{sku\_T9},\ \text{sku\_3081},\ \text{sku\_339},\ \text{sku\_3799},\ \text{sku\_439},\ \text{sku\_539},\ \text{sku\_61399},\ \text{sku\_628},\ \text{sku\_708},\ \text{sku\_77},\ \text{sku\_79},\ \text{sku\_799},\ \text{sku\_8499},\ \text{sku\_89},\ \text{sku\_897},\ \text{sku\_9699},\ \text{sku\_bobo}\}
\]

Define for each product $i\in I$:

- $r_i$: revenue per unit
- $d_i$: demand
- $s_i$: initial inventory
- $x_i$: number of units of product $i$ to fulfill (decision variable, continuous, $0 \leq x_i \leq \min\{d_i, s_i\}$)

Parameters (in source order):

| Product Name   | $r_i$ | $d_i$ | $s_i$ |
|----------------|-------|-------|-------|
| sku_I27        | 238   | 6     | 30    |
| sku_I499       | 287   | 4     | 20    |
| sku_I719       | 268   | 16    | 80    |
| sku_T18        | 318   | 14    | 70    |
| sku_T29        | 207   | 4     | 20    |
| sku_T39        | 258   | 32    | 160   |
| sku_T499       | 249   | 8     | 40    |
| sku_T9         | 227   | 2     | 10    |
| sku_3081       | 198   | 10    | 50    |
| sku_339        | 254   | 8     | 40    |
| sku_3799       | 246   | 18    | 90    |
| sku_439        | 258   | 2     | 10    |
| sku_539        | 268   | 4     | 20    |
| sku_61399      | 278   | 8     | 40    |
| sku_628        | 268   | 2     | 10    |
| sku_708        | 298   | 198   | 990   |
| sku_77         | 258   | 32    | 160   |
| sku_79         | 315   | 18    | 90    |
| sku_799        | 264   | 570   | 2870  |
| sku_8499       | 238   | 6     | 30    |
| sku_89         | 258   | 26    | 130   |
| sku_897        | 268   | 6     | 30    |
| sku_9699       | 288   | 33    | 170   |
| sku_bobo       | 228   | 33    | 170   |

Decision variables:

\[
x_i \in [0, \min\{d_i, s_i\}],\quad \forall i\in I
\]

Objective:

\[
\max \sum_{i\in I} r_i x_i
\]

Subject to:

\[
0 \leq x_i \leq \min\{d_i, s_i\},\quad \forall i\in I
\]

Explicitly, for each product (in source order):

\[
\begin{align*}
0 \leq x_{\text{sku\_I27}} &\leq 6 \\
0 \leq x_{\text{sku\_I499}} &\leq 4 \\
0 \leq x_{\text{sku\_I719}} &\leq 16 \\
0 \leq x_{\text{sku\_T18}} &\leq 14 \\
0 \leq x_{\text{sku\_T29}} &\leq 4 \\
0 \leq x_{\text{sku\_T39}} &\leq 32 \\
0 \leq x_{\text{sku\_T499}} &\leq 8 \\
0 \leq x_{\text{sku\_T9}} &\leq 2 \\
0 \leq x_{\text{sku\_3081}} &\leq 10 \\
0 \leq x_{\text{sku\_339}} &\leq 8 \\
0 \leq x_{\text{sku\_3799}} &\leq 18 \\
0 \leq x_{\text{sku\_439}} &\leq 2 \\
0 \leq x_{\text{sku\_539}} &\leq 4 \\
0 \leq x_{\text{sku\_61399}} &\leq 8 \\
0 \leq x_{\text{sku\_628}} &\leq 2 \\
0 \leq x_{\text{sku\_708}} &\leq 198 \\
0 \leq x_{\text{sku\_77}} &\leq 32 \\
0 \leq x_{\text{sku\_79}} &\leq 18 \\
0 \leq x_{\text{sku\_799}} &\leq 570 \\
0 \leq x_{\text{sku\_8499}} &\leq 6 \\
0 \leq x_{\text{sku\_89}} &\leq 26 \\
0 \leq x_{\text{sku\_897}} &\leq 6 \\
0 \leq x_{\text{sku\_9699}} &\leq 33 \\
0 \leq x_{\text{sku\_bobo}} &\leq 33 \\
\end{align*}
\]

All $x_i$ are continuous and nonnegative.

Maximize total revenue from fulfilled demand, subject to not exceeding demand or initial inventory for each product.