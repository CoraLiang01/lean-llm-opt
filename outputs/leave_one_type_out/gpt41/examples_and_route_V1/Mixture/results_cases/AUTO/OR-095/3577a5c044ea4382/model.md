Let $x_i$ be the number of units of Widget$i$ produced for $i=1,\ldots,141$.
Let $y$ be the amount (kg) of CatalystX sold ($y \geq 0$).
Let $z$ be the amount (kg) of CatalystX disposed ($z \geq 0$).

Parameters (from product_resources.csv and resource_limits.csv):

- For each $i=1,\ldots,141$:
    - $a_i$ = LaborHours per unit (from LaborHours column)
    - $b_i$ = MaterialA per unit (from MaterialA column)
    - $c_i$ = MaterialB per unit (from MaterialB column)
    - $p_i$ = Profit per unit (from Profit column)
- Widget3 generates 5 kg CatalystX per unit produced.
- CatalystX sale price: $300/kg$ (max 1500 kg/month).
- CatalystX disposal cost: $200/kg$.
- Resource limits:
    - LaborHours: 5000
    - MaterialA: 24000
    - MaterialB: 15000

Model:

Objective:
\[
\max \left( \sum_{i=1}^{141} p_i x_i + 300y - 200z \right)
\]

Subject to:

Labor hours:
\[
\sum_{i=1}^{141} a_i x_i \leq 5000
\]

Material A:
\[
\sum_{i=1}^{141} b_i x_i \leq 24000
\]

Material B:
\[
\sum_{i=1}^{141} c_i x_i \leq 15000
\]

CatalystX balance:
\[
5 x_3 = y + z
\]

CatalystX sales cap:
\[
0 \leq y \leq 1500
\]

Nonnegativity and integrality:
\[
x_i \geq 0 \text{ and integer}, \quad \forall i=1,\ldots,141
\]
\[
y \geq 0
\]
\[
z \geq 0
\]

Where the coefficients $a_i$, $b_i$, $c_i$, $p_i$ are as given in product_resources.csv for Widget$i$ ($i=1,\ldots,141$), and $x_3$ is the production quantity of Widget3.

All data is used as retrieved, with all widgets and all resource constraints included.