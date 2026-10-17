Let $x_i \geq 0$ be the number of units produced of Widget $i$ for $i=1,\ldots,141$.

Let $y \geq 0$ be the amount (kg) of CatalystX sold (at most 1500 kg/month).

Let $z \geq 0$ be the amount (kg) of CatalystX disposed.

Parameters (from product_resources.csv):

For each $i=1,\ldots,141$ (Widget$i$):

- $a_i$: labor hours per unit
- $b_i$: MaterialA per unit (kg)
- $c_i$: MaterialB per unit (kg)
- $p_i$: base profit per unit

From resource_limits.csv:

- Total labor hours: $L = 5000$
- Total MaterialA: $A = 24000$
- Total MaterialB: $B = 15000$

CatalystX byproduct: Each unit of Widget3 produces $5$ kg of CatalystX.

CatalystX sales: $300$/kg, up to $1500$ kg/month.

CatalystX disposal: $200$/kg for any unsold CatalystX.

Model:

Maximize total profit:
\[
\max \left(
\sum_{i=1}^{141} p_i x_i
+ 300y
- 200z
\right)
\]

Subject to:
\[
\sum_{i=1}^{141} a_i x_i \leq 5000
\]
\[
\sum_{i=1}^{141} b_i x_i \leq 24000
\]
\[
\sum_{i=1}^{141} c_i x_i \leq 15000
\]
\[
5x_3 = y + z
\]
\[
0 \leq y \leq 1500
\]
\[
x_i \geq 0 \quad \forall i=1,\ldots,141
\]
\[
z \geq 0
\]

Where:

- $x_3$ is the production quantity of Widget3.
- $a_i$, $b_i$, $c_i$, $p_i$ are as given in product_resources.csv for Widget$i$.

All coefficients and identifiers are as retrieved.

#### Retrieved Information

Resource limits:
- LaborHours: 5000
- MaterialA: 24000
- MaterialB: 15000

For each Widget$i$ ($i=1,\ldots,141$), the following parameters are given (examples shown; full list as retrieved):

| Product   | LaborHours | MaterialA | MaterialB | Profit |
|-----------|------------|-----------|-----------|--------|
| Widget1   | 1.6        | 24        | 14        | 525    |
| Widget2   | 2          | 20        | 10        | 678    |
| Widget3   | 2.5        | 12        | 18        | 812    |
| ...       | ...        | ...       | ...       | ...    |
| Widget141 | 1.2        | 11        | 16        | 593    |

CatalystX byproduct: $5x_3$ kg produced, $y$ sold ($0 \leq y \leq 1500$), $z$ disposed ($z \geq 0$), $5x_3 = y + z$.

All variables are continuous and nonnegative.