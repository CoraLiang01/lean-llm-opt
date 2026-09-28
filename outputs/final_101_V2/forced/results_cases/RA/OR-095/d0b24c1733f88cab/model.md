Let $x_i$ be the number of units of Widget$i$ produced for $i=1,\ldots,141$ (where Widget$i$ refers to the product with Product = Widget$i$ in product_resources.csv). Let $y$ be the amount (kg) of CatalystX sold (continuous, $y \geq 0$).

Parameters (from product_resources.csv and resource_limits.csv):

- $a_i$ = LaborHours per unit of Widget$i$
- $b_i$ = MaterialA per unit of Widget$i$
- $c_i$ = MaterialB per unit of Widget$i$
- $p_i$ = Profit per unit of Widget$i$
- LaborHours monthly limit: $5000$
- MaterialA monthly limit: $24000$
- MaterialB monthly limit: $15000$
- Widget3 produces $5$ kg CatalystX per unit produced.
- CatalystX sales price: $300$/kg, sales cap: $1500$ kg/month.
- Unsold CatalystX incurs disposal cost: $200$/kg.

Decision variables:

- $x_i \in \mathbb{Z}_{\geq 0}$, for $i=1,\ldots,141$
- $y \geq 0$ (continuous), amount of CatalystX sold (kg)

Objective function:

\[
\max \left( \sum_{i=1}^{141} p_i x_i + 300y - 200 \left(5x_3 - y\right) \right)
\]

where $x_3$ is the production quantity of Widget3.

Constraints:

1. Labor hours:
\[
\sum_{i=1}^{141} a_i x_i \leq 5000
\]

2. Material A:
\[
\sum_{i=1}^{141} b_i x_i \leq 24000
\]

3. Material B:
\[
\sum_{i=1}^{141} c_i x_i \leq 15000
\]

4. CatalystX sales cannot exceed production:
\[
y \leq 5x_3
\]

5. CatalystX sales cap:
\[
y \leq 1500
\]

6. Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,141
\]
\[
y \geq 0
\]

Parameter values (source order):

Resource limits:
- LaborHours: $5000$
- MaterialA: $24000$
- MaterialB: $15000$

For each $i=1,\ldots,141$ (Widget1 through Widget141), the coefficients $a_i$, $b_i$, $c_i$, $p_i$ are as in the corresponding row of product_resources.csv, e.g.:

- Widget1: $a_1=1.6$, $b_1=24$, $c_1=14$, $p_1=525$
- Widget2: $a_2=2$, $b_2=20$, $c_2=10$, $p_2=678$
- Widget3: $a_3=2.5$, $b_3=12$, $c_3=18$, $p_3=812$
- ...
- Widget141: $a_{141}=1.2$, $b_{141}=11$, $c_{141}=16$, $p_{141}=593$

Complete Model:

\[
\begin{align*}
\max\quad & \sum_{i=1}^{141} p_i x_i + 300y - 200(5x_3 - y) \\
\text{s.t.}\quad
& \sum_{i=1}^{141} a_i x_i \leq 5000 \\
& \sum_{i=1}^{141} b_i x_i \leq 24000 \\
& \sum_{i=1}^{141} c_i x_i \leq 15000 \\
& y \leq 5x_3 \\
& y \leq 1500 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,141 \\
& y \geq 0
\end{align*}
\]

Where all coefficients and identifiers are as retrieved above, in source order.