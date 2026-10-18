Let $x_i$ be the number of units of Widget$i$ produced ($i=1,\ldots,141$), integer and $\geq 0$.

Let $y_{\text{sell}}$ be the amount (kg) of CatalystX sold ($\geq 0$), and $y_{\text{dispose}}$ the amount (kg) of CatalystX disposed ($\geq 0$).

Widget3 generates 5 kg of CatalystX per unit produced.

Parameters (from product_resources.csv and resource_limits.csv):

- For each $i=1,\ldots,141$:
  - $a_i$ = LaborHours per unit (see table)
  - $b_i$ = MaterialA per unit (kg)
  - $c_i$ = MaterialB per unit (kg)
  - $p_i$ = Profit per unit ($)
- LaborHours monthly limit: 5000
- MaterialA monthly limit: 24000 kg
- MaterialB monthly limit: 15000 kg
- CatalystX sales price: $300/kg$ (max 1500 kg/month)
- CatalystX disposal cost: $200/kg$

Explicit data for Widget3: $a_3=2.5$, $b_3=12$, $c_3=18$, $p_3=812$

#### Objective Function

\[
\max \left( \sum_{i=1}^{141} p_i x_i + 300\, y_{\text{sell}} - 200\, y_{\text{dispose}} \right)
\]

#### Constraints

**1. Labor constraint:**
\[
\sum_{i=1}^{141} a_i x_i \leq 5000
\]

**2. Material A constraint:**
\[
\sum_{i=1}^{141} b_i x_i \leq 24000
\]

**3. Material B constraint:**
\[
\sum_{i=1}^{141} c_i x_i \leq 15000
\]

**4. CatalystX balance:**
\[
5 x_3 = y_{\text{sell}} + y_{\text{dispose}}
\]

**5. CatalystX sales cap:**
\[
0 \leq y_{\text{sell}} \leq 1500
\]

**6. Nonnegativity and integrality:**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,141
\]
\[
y_{\text{sell}} \geq 0,\quad y_{\text{dispose}} \geq 0
\]

#### Data Table (first 10 widgets shown for illustration; all 141 widgets included in the model):

| Product    | $a_i$ (LaborHours) | $b_i$ (MaterialA) | $c_i$ (MaterialB) | $p_i$ (Profit) |
|------------|--------------------|-------------------|-------------------|---------------|
| Widget1    | 1.6                | 24                | 14                | 525           |
| Widget2    | 2                  | 20                | 10                | 678           |
| Widget3    | 2.5                | 12                | 18                | 812           |
| Widget4    | 1.9                | 21                | 15                | 769           |
| Widget5    | 0.0                | 15                | 26                | 952           |
| Widget6    | 0.1                | 24                | 17                | 987           |
| Widget7    | 1.2                | 15                | 30                | 644           |
| Widget8    | 1.3                | 21                | 24                | 795           |
| Widget9    | 0.4                | 20                | 30                | 829           |
| Widget10   | 0.9                | 18                | 27                | 574           |
| ...        | ...                | ...               | ...               | ...           |

(All coefficients for $i=1$ to $141$ are as in product_resources.csv, in source order.)

#### Summary

- Decision variables: $x_i$ (integer, $\geq 0$), $y_{\text{sell}}$ (continuous, $0 \leq y_{\text{sell}} \leq 1500$), $y_{\text{dispose}}$ (continuous, $\geq 0$)
- Objective: maximize total profit from widget production and CatalystX sales/disposal
- All resource and byproduct constraints are explicit and source-aligned.