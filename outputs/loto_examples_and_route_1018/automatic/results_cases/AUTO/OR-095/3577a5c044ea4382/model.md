Let $x_i$ be the number of units of Widget $i$ produced, for $i = 1, \ldots, 141$ (with Widget numbers as in the Product column). Let $y$ be the amount (kg) of CatalystX sold (at most 1500 kg). All $x_i \geq 0$ and integer; $y \geq 0$ and continuous.

Widget3 generates 5 kg CatalystX per unit produced. Any CatalystX not sold must be disposed of at a cost.

Parameters (from product_resources.csv and resource_limits.csv):

- $a_i$: LaborHours per unit of Widget $i$
- $b_i$: MaterialA per unit of Widget $i$
- $c_i$: MaterialB per unit of Widget $i$
- $p_i$: Profit per unit of Widget $i$
- LaborHours monthly limit: 5000
- MaterialA monthly limit: 24000
- MaterialB monthly limit: 15000

CatalystX:
- Produced: $5 x_3$ kg
- Sold: $y$ kg, $0 \leq y \leq 1500$
- Unsold: $5 x_3 - y$ kg, must be disposed at $200$/kg
- Sale price: $300$/kg

Objective:
\[
\max \left( \sum_{i=1}^{141} p_i x_i + 300 y - 200 (5 x_3 - y) \right)
\]
or equivalently,
\[
\max \left( \sum_{i=1}^{141} p_i x_i - 100 \cdot 5 x_3 + 500 y \right)
\]
since $300y - 200(5x_3 - y) = -100 \cdot 5 x_3 + 500 y$.

Subject to:

1. Labor constraint:
\[
\sum_{i=1}^{141} a_i x_i \leq 5000
\]

2. Material A constraint:
\[
\sum_{i=1}^{141} b_i x_i \leq 24000
\]

3. Material B constraint:
\[
\sum_{i=1}^{141} c_i x_i \leq 15000
\]

4. CatalystX sales cap:
\[
0 \leq y \leq \min(5 x_3, 1500)
\]

5. Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,141
\]
\[
y \geq 0
\]

Explicitly, for $i=1,\ldots,141$, the coefficients $a_i$, $b_i$, $c_i$, $p_i$ are as in the retrieved product_resources.csv, in source order:

| Product   | $a_i$ (LaborHours) | $b_i$ (MaterialA) | $c_i$ (MaterialB) | $p_i$ (Profit) |
|-----------|--------------------|-------------------|-------------------|---------------|
| Widget1   | 1.6                | 24                | 14                | 525           |
| Widget2   | 2                  | 20                | 10                | 678           |
| Widget3   | 2.5                | 12                | 18                | 812           |
| Widget4   | 1.9                | 21                | 15                | 769           |
| Widget5   | 0.0                | 15                | 26                | 952           |
| ...       | ...                | ...               | ...               | ...           |
| Widget141 | 1.2                | 11                | 16                | 593           |

(Continue for all 141 widgets as in the retrieved data.)

Full model:

\[
\begin{align*}
\max\ & \sum_{i=1}^{141} p_i x_i - 500 x_3 + 500 y \\
\text{s.t.} \\
& \sum_{i=1}^{141} a_i x_i \leq 5000 \\
& \sum_{i=1}^{141} b_i x_i \leq 24000 \\
& \sum_{i=1}^{141} c_i x_i \leq 15000 \\
& 0 \leq y \leq 5 x_3 \\
& y \leq 1500 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,141 \\
& y \geq 0
\end{align*}
\]

Where all coefficients and identifiers are as in the retrieved CSVs, in original order.