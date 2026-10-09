Let $x_i$ denote the number of units of Widget $i$ to produce, for $i = 1, \ldots, 141$ (with Widget3 corresponding to $i=3$). Let $y$ denote the amount (kg) of CatalystX sold (continuous, $0 \leq y \leq 1500$). All $x_i \geq 0$ and continuous (unless otherwise specified).

Parameters (from product_resources.csv and resource_limits.csv):

- $a_i$: Labor hours per unit of Widget $i$
- $b_i$: Material A (kg) per unit of Widget $i$
- $c_i$: Material B (kg) per unit of Widget $i$
- $p_i$: Base profit per unit of Widget $i$
- $L = 5000$: Monthly labor hour limit
- $A = 24000$: Monthly Material A limit (kg)
- $B = 15000$: Monthly Material B limit (kg)
- Widget3 produces 5 kg CatalystX per unit produced
- CatalystX sale price: $300$/kg, max $1500$ kg/month
- CatalystX disposal cost: $200$/kg for any unsold

Decision variables:
- $x_i \geq 0$ (continuous), for $i = 1, \ldots, 141$
- $y \geq 0$ (continuous), amount of CatalystX sold, $y \leq 1500$

Objective function:

\[
\max \left[
\sum_{i=1}^{141} p_i x_i
+ 300y
- 200 \left(5x_3 - y\right)
\right]
\]

subject to:

\[
\begin{align*}
&\sum_{i=1}^{141} a_i x_i \leq 5000 \\
&\sum_{i=1}^{141} b_i x_i \leq 24000 \\
&\sum_{i=1}^{141} c_i x_i \leq 15000 \\
&y \leq 5x_3 \\
&y \leq 1500 \\
&x_i \geq 0 \quad \forall i=1,\ldots,141 \\
&y \geq 0
\end{align*}
\]

Where:

- $a_i$, $b_i$, $c_i$, $p_i$ are as given in product_resources.csv for Widget$i$.
- $x_3$ is the production quantity of Widget3.
- $5x_3$ is the total kg of CatalystX generated.
- $y$ is the amount of CatalystX sold (cannot exceed $5x_3$ or $1500$ kg).
- Any unsold CatalystX ($5x_3 - y$) incurs a disposal cost of $200$/kg.

Explicitly, for $i=1$ to $141$:

| Widget | $a_i$ (LaborHours) | $b_i$ (MaterialA) | $c_i$ (MaterialB) | $p_i$ (Profit) |
|--------|--------------------|-------------------|-------------------|---------------|
| Widget1 | 1.6 | 24 | 14 | 525 |
| Widget2 | 2 | 20 | 10 | 678 |
| Widget3 | 2.5 | 12 | 18 | 812 |
| Widget4 | 1.9 | 21 | 15 | 769 |
| ... | ... | ... | ... | ... |
| Widget141 | 1.2 | 11 | 16 | 593 |

(Continue for all widgets as per the retrieved data.)

Summary of constraints:

- Labor: $\sum_{i=1}^{141} a_i x_i \leq 5000$
- Material A: $\sum_{i=1}^{141} b_i x_i \leq 24000$
- Material B: $\sum_{i=1}^{141} c_i x_i \leq 15000$
- CatalystX sales: $y \leq 5x_3$, $y \leq 1500$
- Nonnegativity: $x_i \geq 0$ for all $i$, $y \geq 0$

Variable domains:

- $x_i \geq 0$ (continuous, unless integer production is required)
- $y \geq 0$ (continuous)

All coefficients and limits are as given in the retrieved CSV data.