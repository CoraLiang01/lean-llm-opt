Let $x_i$ be the number of units produced of Widget $i$ for $i = 1, \ldots, 141$ (where $i$ indexes the widgets in the order and with the names as given in product_resources.csv). Let $y$ be the amount (kg) of CatalystX sold (continuous, $y \geq 0$).

Parameters (from product_resources.csv, in source order):

- For each widget $i$:
  - $l_i$ = LaborHours per unit
  - $a_i$ = MaterialA per unit (kg)
  - $b_i$ = MaterialB per unit (kg)
  - $p_i$ = Profit per unit

Special byproduct:
- Widget3 produces 5 kg CatalystX per unit produced: $5 x_3$ kg total CatalystX generated.
- CatalystX can be sold at $300$/kg, up to $1500$ kg/month: $0 \leq y \leq 1500$.
- Unsold CatalystX must be disposed at $200$/kg.

Resource limits (from resource_limits.csv):

- $\sum_{i=1}^{141} l_i x_i \leq 5000$ (LaborHours)
- $\sum_{i=1}^{141} a_i x_i \leq 24000$ (MaterialA)
- $\sum_{i=1}^{141} b_i x_i \leq 15000$ (MaterialB)

Objective function:

Maximize total profit:
\[
\max \left\{
\sum_{i=1}^{141} p_i x_i
+ 300y
- 200 \left(5x_3 - y\right)
\right\}
\]
where $5x_3$ is the total CatalystX generated, $y$ is the amount sold, and $5x_3 - y$ is the amount disposed.

Subject to:

1. Labor constraint:
   \[
   \sum_{i=1}^{141} l_i x_i \leq 5000
   \]
2. Material A constraint:
   \[
   \sum_{i=1}^{141} a_i x_i \leq 24000
   \]
3. Material B constraint:
   \[
   \sum_{i=1}^{141} b_i x_i \leq 15000
   \]
4. CatalystX sales cannot exceed production or market cap:
   \[
   0 \leq y \leq \min\{5x_3, 1500\}
   \]
   (Implemented as two constraints:)
   \[
   y \leq 5x_3
   \]
   \[
   y \leq 1500
   \]
5. Nonnegativity and integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,141
   \]
   \[
   y \geq 0
   \]

Explicitly, using the data as retrieved (showing the first few widgets for illustration; all 141 widgets are included in the model):

For $i=1$ to $141$ (Widget1 to Widget141, in the order and with the coefficients as in product_resources.csv):

- $l_i$ = LaborHours (e.g., Widget1: $l_1 = 1.6$)
- $a_i$ = MaterialA (e.g., Widget1: $a_1 = 24$)
- $b_i$ = MaterialB (e.g., Widget1: $b_1 = 14$)
- $p_i$ = Profit (e.g., Widget1: $p_1 = 525$)

The complete model:

\[
\begin{align*}
\max \quad & \sum_{i=1}^{141} p_i x_i + 300y - 200(5x_3 - y) \\
\text{s.t.} \quad
& \sum_{i=1}^{141} l_i x_i \leq 5000 \\
& \sum_{i=1}^{141} a_i x_i \leq 24000 \\
& \sum_{i=1}^{141} b_i x_i \leq 15000 \\
& y \leq 5x_3 \\
& y \leq 1500 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,141 \\
& y \geq 0
\end{align*}
\]

Where all coefficients $l_i$, $a_i$, $b_i$, $p_i$ are as given in product_resources.csv, in the original file order.

All variables and constraints are indexed and aligned by the explicit Product names and resource names as retrieved. No data or constraints are omitted.