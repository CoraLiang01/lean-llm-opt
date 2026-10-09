Let $x_i$ be the integer number of contracts (positive for buy/long, negative for sell/short) for option $i$, $i=1,\dots,120$.

Let $y_i \geq 0$ be an auxiliary variable representing $|x_i|$ for each $i$.

Let $A_{i,j}$ be the binary indicator that option $i$ references asset $j$, $j=1,\dots,6$.

Let $\text{Cost}_i$, $\Delta_i$, $\Gamma_i$, $\text{Vega}_i$, $\text{MaxLong}_i$, $\text{MaxShort}_i$ be the per-option coefficients from OptionCharacteristics.csv.

Let the initial exposures and tolerances be:
- $\Delta_{\text{init}} = 0.25$, $\Gamma_{\text{init}} = 0.08$, $\text{Vega}_{\text{init}} = 0.17$
- $\text{Tolerance}_\Delta = 0.06$, $\text{Tolerance}_\Gamma = 0.05$, $\text{Tolerance}_\text{Vega} = 0.07$

The model is:

Minimize total hedging cost:
$$
\min \sum_{i=1}^{120} \text{Cost}_i \cdot y_i
$$

Subject to:

For each $i=1,\dots,120$:
\[
\text{MaxShort}_i \leq x_i \leq \text{MaxLong}_i
\]
\[
y_i \geq x_i
\]
\[
y_i \geq -x_i
\]
\[
y_i \geq 0
\]
\[
x_i \in \mathbb{Z}
\]

For each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$:
\[
\left| G_{\text{init}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G_i \cdot A_{i,j} \cdot x_i \right| \leq \text{Tolerance}_G
\]
That is, for each Greek $G$:
\[
- \text{Tolerance}_G \leq G_{\text{init}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G_i \cdot A_{i,j} \cdot x_i \leq \text{Tolerance}_G
\]

Where:
- $G_{\text{init}}$ is the initial exposure for Greek $G$ (see above).
- $G_i$ is the per-contract Greek exposure of option $i$ (from OptionCharacteristics.csv).
- $A_{i,j}$ is from Option_AssetReferenceMatrix.csv.

All coefficients and bounds are as given in the retrieved data, preserving original row and identifier order.

---

**Data used (first few rows for illustration):**

From OptionCharacteristics.csv (original order):
| Option   | Cost | Delta  | Gamma | Vega  | MaxLong | MaxShort |
|----------|------|--------|-------|-------|---------|----------|
| Opt_1    | 9    | -0.54  | 0.12  | 0.1   | 9       | -14      |
| Opt_2    | 6    | 0.51   | 0.1   | 0.16  | 9       | -14      |
| ...      | ...  | ...    | ...   | ...   | ...     | ...      |

From Option_AssetReferenceMatrix.csv (original order):
| Option   | Asset_1 | Asset_2 | Asset_3 | Asset_4 | Asset_5 | Asset_6 |
|----------|---------|---------|---------|---------|---------|---------|
| Opt_1    | 0       | 0       | 0       | 1       | 0       | 0       |
| Opt_2    | 1       | 1       | 0       | 0       | 0       | 0       |
| ...      | ...     | ...     | ...     | ...     | ...     | ...     |

**Constants:**
- $\Delta_{\text{init}} = 0.25$, $\Gamma_{\text{init}} = 0.08$, $\text{Vega}_{\text{init}} = 0.17$
- $\text{Tolerance}_\Delta = 0.06$, $\text{Tolerance}_\Gamma = 0.05$, $\text{Tolerance}_\text{Vega} = 0.07$

**Decision variables:**
- $x_i \in \mathbb{Z}$, $\text{MaxShort}_i \leq x_i \leq \text{MaxLong}_i$
- $y_i \geq |x_i|$

**Objective:**
- Minimize $\sum_{i=1}^{120} \text{Cost}_i \cdot y_i$

**Risk constraints:**
- For each $G \in \{\Delta, \Gamma, \text{Vega}\}$:
  $$
  -\text{Tolerance}_G \leq G_{\text{init}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G_i \cdot A_{i,j} \cdot x_i \leq \text{Tolerance}_G
  $$

All coefficients, bounds, and indices are as in the retrieved data, in original order.