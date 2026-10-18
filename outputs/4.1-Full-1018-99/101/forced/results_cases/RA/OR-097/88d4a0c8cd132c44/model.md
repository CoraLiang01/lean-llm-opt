Let $x_i$ be the integer number of contracts (positive for buy/long, negative for sell/short) for option $i$, $i=1,\dots,120$ (with option identifiers as in OptionCharacteristics.csv). Introduce auxiliary variables $y_i \geq 0$ for $i=1,\dots,120$ such that $y_i \geq x_i$ and $y_i \geq -x_i$ to model $|x_i|$ in the objective.

**Parameters (from data):**

- For each option $i$ (OptionCharacteristics.csv, source order):
    - Option: $\text{Opt}_i$
    - $\text{Cost}_i$, $\text{Delta}_i$, $\text{Gamma}_i$, $\text{Vega}_i$, $\text{MaxLong}_i$, $\text{MaxShort}_i$
- For each option $i$ and asset $j$ (Option_AssetReferenceMatrix.csv, source order):
    - $A_{i,j} = 1$ if option $i$ references asset $j$, $0$ otherwise; $j=1,\dots,6$
- Initial exposures:
    - $\Delta_{\text{initial}} = 0.25$
    - $\Gamma_{\text{initial}} = 0.08$
    - $\text{Vega}_{\text{initial}} = 0.17$
- Tolerances:
    - $|\Delta| \leq 0.06$
    - $|\Gamma| \leq 0.05$
    - $|\text{Vega}| \leq 0.07$

---

**Model:**

Minimize total hedging cost:
$$
\min \sum_{i=1}^{120} \text{Cost}_i \cdot y_i
$$

Subject to:

For each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$:
\[
\left| G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G_i \cdot A_{i,j} \cdot x_i \right| \leq \text{Tolerance}_G
\]
That is, for each $G$:
\[
- \text{Tolerance}_G \leq G_{\text{initial}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G_i \cdot A_{i,j} \cdot x_i \leq \text{Tolerance}_G
\]

Where:
- For Delta: $G_i = \text{Delta}_i$, $G_{\text{initial}} = 0.25$, $\text{Tolerance}_G = 0.06$
- For Gamma: $G_i = \text{Gamma}_i$, $G_{\text{initial}} = 0.08$, $\text{Tolerance}_G = 0.05$
- For Vega: $G_i = \text{Vega}_i$, $G_{\text{initial}} = 0.17$, $\text{Tolerance}_G = 0.07$

Trading limits for each option $i$:
\[
\text{MaxShort}_i \leq x_i \leq \text{MaxLong}_i \qquad \forall i=1,\dots,120
\]

Absolute value modeling for each $i$:
\[
y_i \geq x_i
\]
\[
y_i \geq -x_i
\]
\[
y_i \geq 0
\]

Variable domains:
\[
x_i \in \mathbb{Z}, \quad y_i \geq 0 \qquad \forall i=1,\dots,120
\]

---

**Data (source order, all rows):**

- OptionCharacteristics.csv: For $i=1,\dots,120$, each row provides Option, Cost, Delta, Gamma, Vega, MaxLong, MaxShort.
- Option_AssetReferenceMatrix.csv: For $i=1,\dots,120$, each row provides Option (as Unnamed: 0), Asset_1, ..., Asset_6 (binary).

All coefficients and identifiers are as in the retrieved data, in original file and row order. No data omitted.

---

**Complete Model (with explicit indices):**

Minimize:
\[
\sum_{i=1}^{120} \text{Cost}_i \cdot y_i
\]

Subject to:
\[
-0.06 \leq 0.25 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Delta}_i \cdot A_{i,j} \cdot x_i \leq 0.06
\]
\[
-0.05 \leq 0.08 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Gamma}_i \cdot A_{i,j} \cdot x_i \leq 0.05
\]
\[
-0.07 \leq 0.17 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}_i \cdot A_{i,j} \cdot x_i \leq 0.07
\]
\[
\text{MaxShort}_i \leq x_i \leq \text{MaxLong}_i \qquad \forall i=1,\dots,120
\]
\[
y_i \geq x_i,\quad y_i \geq -x_i,\quad y_i \geq 0 \qquad \forall i=1,\dots,120
\]
\[
x_i \in \mathbb{Z},\quad y_i \geq 0 \qquad \forall i=1,\dots,120
\]

All coefficients and identifiers are as in the retrieved data, in original file and row order.