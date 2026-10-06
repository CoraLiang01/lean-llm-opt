Let $x_i$ denote the number of units of Widget$i$ produced for $i=1,\ldots,141$.
Let $y$ denote the amount (kg) of CatalystX sold ($y \geq 0$).

Parameters (from product_resources.csv and resource_limits.csv):

- For each $i=1,\ldots,141$:
    - $l_i$ = LaborHours per unit of Widget$i$
    - $a_i$ = MaterialA per unit of Widget$i$
    - $b_i$ = MaterialB per unit of Widget$i$
    - $p_i$ = Profit per unit of Widget$i$
- Widget3 generates 5 kg CatalystX per unit produced.
- CatalystX can be sold at $300/kg$ (up to 1500 kg/month), or disposed at $200/kg$.
- Monthly limits: LaborHours $\leq 5000$, MaterialA $\leq 24000$, MaterialB $\leq 15000$.

The complete model:

Objective:
\[
\max \left( \sum_{i=1}^{141} p_i x_i + 300y - 200 \left(5x_3 - y\right) \right)
\]
where $5x_3$ is the total kg of CatalystX generated, $y$ is the amount sold, and $5x_3 - y$ is the amount disposed.

Subject to:
\[
\sum_{i=1}^{141} l_i x_i \leq 5000
\]
\[
\sum_{i=1}^{141} a_i x_i \leq 24000
\]
\[
\sum_{i=1}^{141} b_i x_i \leq 15000
\]
\[
y \leq 5x_3
\]
\[
y \leq 1500
\]
\[
x_i \geq 0 \text{ and integer}, \quad \forall i=1,\ldots,141
\]
\[
y \geq 0
\]

Where the coefficients are as follows (source order preserved):

Resource limits:
- LaborHours: 5000
- MaterialA: 24000
- MaterialB: 15000

Product parameters (partial, full list as in product_resources.csv):

| Product   | LaborHours | MaterialA | MaterialB | Profit |
|-----------|------------|-----------|-----------|--------|
| Widget1   | 1.6        | 24        | 14        | 525    |
| Widget2   | 2          | 20        | 10        | 678    |
| Widget3   | 2.5        | 12        | 18        | 812    |
| Widget4   | 1.9        | 21        | 15        | 769    |
| ...       | ...        | ...       | ...       | ...    |
| Widget141 | 1.2        | 11        | 16        | 593    |

All coefficients and identifiers are as in the retrieved data.

Decision variables:
- $x_i$: integer, $\geq 0$, for $i=1,\ldots,141$
- $y$: continuous, $\geq 0$

Complete model:

\[
\begin{align*}
\max\ & \sum_{i=1}^{141} p_i x_i + 300y - 200(5x_3 - y) \\
\text{s.t.}\quad
& \sum_{i=1}^{141} l_i x_i \leq 5000 \\
& \sum_{i=1}^{141} a_i x_i \leq 24000 \\
& \sum_{i=1}^{141} b_i x_i \leq 15000 \\
& y \leq 5x_3 \\
& y \leq 1500 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,141 \\
& y \geq 0
\end{align*}
\]

Where all $l_i$, $a_i$, $b_i$, $p_i$ are as in product_resources.csv, in the original order.