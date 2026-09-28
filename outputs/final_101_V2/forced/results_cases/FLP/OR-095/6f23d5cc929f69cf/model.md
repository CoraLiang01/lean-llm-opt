##### Decision Variables

Let $x_i \geq 0$ denote the number of units produced of Widget $i$, for $i = 1,2,\ldots,141$.

Let $s \geq 0$ denote the kilograms of CatalystX sold (at most 1500 kg).

Let $w \geq 0$ denote the kilograms of CatalystX disposed as hazardous waste.

##### Parameters

For each widget $i$ ($i=1,\ldots,141$):

- $l_i$: labor hours required per unit (from product_resources.csv)
- $a_i$: Material A required per unit (kg)
- $b_i$: Material B required per unit (kg)
- $p_i$: base profit per unit

CatalystX is generated only by Widget3 at a rate of 5 kg per unit produced.

Resource limits (from resource_limits.csv):

- Total labor hours: $L = 5000$
- Total Material A: $A = 24000$
- Total Material B: $B = 15000$

CatalystX:

- Sale price: $300$/kg (up to 1500 kg/month)
- Disposal cost: $200$/kg (for any unsold amount)
- Generation: $5$ kg per unit of Widget3 produced

##### Objective Function

Maximize total profit:
\[
\max \left[
\sum_{i=1}^{141} p_i x_i
+ 300 s
- 200 w
\right]
\]

##### Constraints

1. **Labor hours constraint:**
   \[
   \sum_{i=1}^{141} l_i x_i \leq 5000
   \]
2. **Material A constraint:**
   \[
   \sum_{i=1}^{141} a_i x_i \leq 24000
   \]
3. **Material B constraint:**
   \[
   \sum_{i=1}^{141} b_i x_i \leq 15000
   \]
4. **CatalystX mass balance:**
   \[
   5 x_3 = s + w
   \]
   (where $x_3$ is the production quantity of Widget3)
5. **CatalystX sales cap:**
   \[
   0 \leq s \leq 1500
   \]
6. **Nonnegativity:**
   \[
   x_i \geq 0 \quad \forall i=1,\ldots,141
   \]
   \[
   s \geq 0,\quad w \geq 0
   \]

##### Parameters from product_resources.csv

For $i=1,\ldots,141$ (Widget1 to Widget141):

| $i$ | Product    | $l_i$ (LaborHours) | $a_i$ (MaterialA) | $b_i$ (MaterialB) | $p_i$ (Profit) |
|-----|------------|--------------------|-------------------|-------------------|---------------|
| 1   | Widget1    | 1.6                | 24                | 14                | 525           |
| 2   | Widget2    | 2                  | 20                | 10                | 678           |
| 3   | Widget3    | 2.5                | 12                | 18                | 812           |
| 4   | Widget4    | 1.9                | 21                | 15                | 769           |
| ... | ...        | ...                | ...               | ...               | ...           |
| 141 | Widget141  | 1.2                | 11                | 16                | 593           |

(Full table as retrieved above.)

##### Complete Mathematical Model

\[
\begin{align*}
\max_{x_i,\,s,\,w} \quad & \sum_{i=1}^{141} p_i x_i + 300 s - 200 w \\
\text{s.t.} \quad
& \sum_{i=1}^{141} l_i x_i \leq 5000 \\
& \sum_{i=1}^{141} a_i x_i \leq 24000 \\
& \sum_{i=1}^{141} b_i x_i \leq 15000 \\
& 5 x_3 = s + w \\
& 0 \leq s \leq 1500 \\
& x_i \geq 0 \quad \forall i=1,\ldots,141 \\
& w \geq 0
\end{align*}
\]

where all $l_i$, $a_i$, $b_i$, $p_i$ are as given in product_resources.csv for Widget1 through Widget141.