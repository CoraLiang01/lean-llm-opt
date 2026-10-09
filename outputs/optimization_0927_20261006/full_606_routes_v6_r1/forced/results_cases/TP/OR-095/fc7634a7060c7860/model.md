Let $x_p \geq 0$ be the number of units produced of widget $p$ for each $p \in \{\text{Widget1},\ldots,\text{Widget141}\}$.

Let $y \geq 0$ be the kilograms of CatalystX sold (continuous).

Parameters (from product_resources.csv, in source order):

For each widget $p$:
- $\text{LaborHours}_p$ = labor hours per unit
- $\text{MaterialA}_p$ = kg Material A per unit
- $\text{MaterialB}_p$ = kg Material B per unit
- $\text{Profit}_p$ = base profit per unit

CatalystX byproduct: Each unit of Widget3 produces 5 kg CatalystX.

CatalystX sales price: $300$/kg, up to 1500 kg/month.

CatalystX disposal cost: $200$/kg for any unsold CatalystX.

Resource limits (from resource_limits.csv):
- Total labor hours: $\leq 5000$
- Total Material A: $\leq 24000$ kg
- Total Material B: $\leq 15000$ kg

Objective:
\[
\max \left(
\sum_{p=1}^{141} \text{Profit}_p \cdot x_p
+ 300y
- 200 \left(5x_3 - y\right)
\right)
\]
where $x_3$ is the production quantity of Widget3.

Constraints:
\[
\sum_{p=1}^{141} \text{LaborHours}_p \cdot x_p \leq 5000
\]
\[
\sum_{p=1}^{141} \text{MaterialA}_p \cdot x_p \leq 24000
\]
\[
\sum_{p=1}^{141} \text{MaterialB}_p \cdot x_p \leq 15000
\]
\[
0 \leq y \leq 5x_3
\]
\[
y \leq 1500
\]
\[
x_p \geq 0 \quad \forall p=1,\ldots,141
\]
\[
y \geq 0
\]

Where the coefficients for each widget $p$ are as follows (source order):

| $p$ | Product   | LaborHours | MaterialA | MaterialB | Profit |
|-----|-----------|------------|-----------|-----------|--------|
| 1   | Widget1   | 1.6        | 24        | 14        | 525    |
| 2   | Widget2   | 2          | 20        | 10        | 678    |
| 3   | Widget3   | 2.5        | 12        | 18        | 812    |
| 4   | Widget4   | 1.9        | 21        | 15        | 769    |
| 5   | Widget5   | 0.0        | 15        | 26        | 952    |
| 6   | Widget6   | 0.1        | 24        | 17        | 987    |
| 7   | Widget7   | 1.2        | 15        | 30        | 644    |
| 8   | Widget8   | 1.3        | 21        | 24        | 795    |
| 9   | Widget9   | 0.4        | 20        | 30        | 829    |
| 10  | Widget10  | 0.9        | 18        | 27        | 574    |
| ... | ...       | ...        | ...       | ...       | ...    |
| 141 | Widget141 | 1.2        | 11        | 16        | 593    |

(Full table continues for all 141 widgets as retrieved.)

Summary of retrieved information:

- Resource limits:
    - LaborHours: 5000
    - MaterialA: 24000
    - MaterialB: 15000
- Widget parameters: For each Widget1–Widget141, as above.
- Widget3: produces 5 kg CatalystX per unit.
- CatalystX: sell up to 1500 kg at $300/kg; unsold incurs $200/kg disposal.

Decision variables:
- $x_p \geq 0$ for $p=1,\ldots,141$
- $y \geq 0$

Maximize:
\[
\sum_{p=1}^{141} \text{Profit}_p \cdot x_p + 300y - 200(5x_3 - y)
\]

Subject to:
\[
\sum_{p=1}^{141} \text{LaborHours}_p \cdot x_p \leq 5000
\]
\[
\sum_{p=1}^{141} \text{MaterialA}_p \cdot x_p \leq 24000
\]
\[
\sum_{p=1}^{141} \text{MaterialB}_p \cdot x_p \leq 15000
\]
\[
0 \leq y \leq 5x_3
\]
\[
y \leq 1500
\]
\[
x_p \geq 0 \quad \forall p
\]
\[
y \geq 0
\]