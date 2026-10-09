##### Decision Variables

- $x_p \geq 0$: Number of units produced of widget $p$, for all $p \in \mathcal{P} = \{\text{Widget1}, \ldots, \text{Widget141}\}$.
- $s \geq 0$: Amount (kg) of CatalystX sold (market cap: $s \leq 1500$).
- $w \geq 0$: Amount (kg) of CatalystX disposed as hazardous waste.

##### Parameters

Let the following vectors/matrices be defined for $p \in \mathcal{P}$:

- $\text{LaborHours}_p$: Labor hours required per unit of widget $p$.
- $\text{MaterialA}_p$: Material A required per unit of widget $p$.
- $\text{MaterialB}_p$: Material B required per unit of widget $p$.
- $\text{Profit}_p$: Base profit per unit of widget $p$.

Resource limits:

- Total labor hours available: $5000$
- Total Material A available: $24000$ kg
- Total Material B available: $15000$ kg

CatalystX byproduct:

- Only Widget3 produces CatalystX, at $5$ kg per unit produced.
- CatalystX can be sold at $300$ per kg, up to $1500$ kg per month.
- Any unsold CatalystX must be disposed at $200$ per kg.

##### Objective Function

\[
\max \left(
    \sum_{p \in \mathcal{P}} \text{Profit}_p \cdot x_p
    + 300 s
    - 200 w
\right)
\]

##### Constraints

1. **Labor hours constraint:**
   \[
   \sum_{p \in \mathcal{P}} \text{LaborHours}_p \cdot x_p \leq 5000
   \]
2. **Material A constraint:**
   \[
   \sum_{p \in \mathcal{P}} \text{MaterialA}_p \cdot x_p \leq 24000
   \]
3. **Material B constraint:**
   \[
   \sum_{p \in \mathcal{P}} \text{MaterialB}_p \cdot x_p \leq 15000
   \]
4. **CatalystX mass balance:**
   \[
   5 x_{\text{Widget3}} = s + w
   \]
5. **CatalystX sales cap:**
   \[
   0 \leq s \leq 1500
   \]
6. **Nonnegativity:**
   \[
   x_p \geq 0 \quad \forall p \in \mathcal{P}
   \]
   \[
   s \geq 0,\quad w \geq 0
   \]

##### Full Parameter Table (first 10 widgets shown; full list continues through Widget141):

| Product   | LaborHours | MaterialA | MaterialB | Profit |
|-----------|------------|-----------|-----------|--------|
| Widget1   | 1.6        | 24        | 14        | 525    |
| Widget2   | 2          | 20        | 10        | 678    |
| Widget3   | 2.5        | 12        | 18        | 812    |
| Widget4   | 1.9        | 21        | 15        | 769    |
| Widget5   | 0.0        | 15        | 26        | 952    |
| Widget6   | 0.1        | 24        | 17        | 987    |
| Widget7   | 1.2        | 15        | 30        | 644    |
| Widget8   | 1.3        | 21        | 24        | 795    |
| Widget9   | 0.4        | 20        | 30        | 829    |
| Widget10  | 0.9        | 18        | 27        | 574    |
| ...       | ...        | ...       | ...       | ...    |
| Widget141 | 1.2        | 11        | 16        | 593    |

(For all $p \in \mathcal{P}$, use the values as retrieved above.)

##### Summary of Model

- Decide $x_p$ for all widgets, $s$ (CatalystX sold), $w$ (CatalystX disposed).
- Maximize total profit from widgets and CatalystX sales, minus disposal cost.
- Respect labor, Material A, and Material B limits.
- Enforce CatalystX mass balance and sales cap.

##### Retrieved Information

- Resource limits:
  - LaborHours: 5000
  - MaterialA: 24000
  - MaterialB: 15000
- Widget parameters (all 141 widgets, see above for first 10; full list as in product_resources.csv).
- Widget3 produces 5 kg CatalystX per unit.
- CatalystX: sell at $300/kg$ (max 1500 kg/month), dispose at $200/kg$.

---

**Variables:**
- $x_p \geq 0$ for $p = \text{Widget1}, \ldots, \text{Widget141}$
- $s \geq 0$, $w \geq 0$

**Parameters:**
- $\text{LaborHours}_p$, $\text{MaterialA}_p$, $\text{MaterialB}_p$, $\text{Profit}_p$ as above.

**Objective:**
\[
\max \left(
    \sum_{p \in \mathcal{P}} \text{Profit}_p \cdot x_p
    + 300 s
    - 200 w
\right)
\]

**Subject to:**
\[
\sum_{p \in \mathcal{P}} \text{LaborHours}_p \cdot x_p \leq 5000
\]
\[
\sum_{p \in \mathcal{P}} \text{MaterialA}_p \cdot x_p \leq 24000
\]
\[
\sum_{p \in \mathcal{P}} \text{MaterialB}_p \cdot x_p \leq 15000
\]
\[
5 x_{\text{Widget3}} = s + w
\]
\[
0 \leq s \leq 1500
\]
\[
x_p \geq 0 \quad \forall p \in \mathcal{P}
\]
\[
s \geq 0,\quad w \geq 0
\]