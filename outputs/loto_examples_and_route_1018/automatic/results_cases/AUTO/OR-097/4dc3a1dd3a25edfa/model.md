Let $x_i$ be the integer number of contracts (positive for buy/long, negative for sell/short) for option $i$, $i=1,\dots,120$.

Let $y_i \geq 0$ be an auxiliary variable representing $|x_i|$ for each $i$.

Let $A[i,j]$ be the binary indicator that option $i$ references asset $j$, $j=1,\dots,6$.

Let $\text{Cost}[i]$, $\text{Delta}[i]$, $\text{Gamma}[i]$, $\text{Vega}[i]$, $\text{MaxLong}[i]$, $\text{MaxShort}[i]$ be the coefficients from OptionCharacteristics.csv, for $i=1,\dots,120$.

Let $\Delta_{\text{init}} = 0.25$, $\Gamma_{\text{init}} = 0.08$, $\text{Vega}_{\text{init}} = 0.17$.

Let $\text{Tolerance}_\Delta = 0.06$, $\text{Tolerance}_\Gamma = 0.05$, $\text{Tolerance}_\text{Vega} = 0.07$.

The model is:

Minimize total hedging cost:
$$
\min \sum_{i=1}^{120} \text{Cost}[i] \cdot y_i
$$

Subject to:

For each Greek $G \in \{\Delta, \Gamma, \text{Vega}\}$,
\[
\left| G_{\text{init}} + \sum_{i=1}^{120} \sum_{j=1}^{6} G[i] \cdot A[i,j] \cdot x_i \right| \leq \text{Tolerance}_G
\]
That is, explicitly:

Delta constraint:
\[
-0.06 \leq 0.25 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Delta}[i] \cdot A[i,j] \cdot x_i \leq 0.06
\]

Gamma constraint:
\[
-0.05 \leq 0.08 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Gamma}[i] \cdot A[i,j] \cdot x_i \leq 0.05
\]

Vega constraint:
\[
-0.07 \leq 0.17 + \sum_{i=1}^{120} \sum_{j=1}^{6} \text{Vega}[i] \cdot A[i,j] \cdot x_i \leq 0.07
\]

Trading limits for each option:
\[
\text{MaxShort}[i] \leq x_i \leq \text{MaxLong}[i], \quad i=1,\dots,120
\]

Absolute value constraints for cost:
\[
y_i \geq x_i,\quad y_i \geq -x_i,\quad y_i \geq 0,\quad i=1,\dots,120
\]

Integrality:
\[
x_i \in \mathbb{Z},\quad i=1,\dots,120
\]

All coefficients and option-asset references are as given in the retrieved OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv, preserving their original order and identifiers.

---

**Retrieved Data (abbreviated for illustration; full data used in model):**

From OptionCharacteristics.csv (first 3 rows shown, all 120 used):

| Option  | Cost | Delta  | Gamma | Vega  | MaxLong | MaxShort |
|---------|------|--------|-------|-------|---------|----------|
| Opt_1   | 9    | -0.54  | 0.12  | 0.10  | 9       | -14      |
| Opt_2   | 6    | 0.51   | 0.10  | 0.16  | 9       | -14      |
| Opt_3   | 13   | 0.17   | 0.02  | 0.19  | 10      | -7       |
| ...     | ...  | ...    | ...   | ...   | ...     | ...      |

From Option_AssetReferenceMatrix.csv (first 3 rows shown, all 120 used):

| Option  | Asset_1 | Asset_2 | Asset_3 | Asset_4 | Asset_5 | Asset_6 |
|---------|---------|---------|---------|---------|---------|---------|
| Opt_1   | 0       | 0       | 0       | 1       | 0       | 0       |
| Opt_2   | 1       | 1       | 0       | 0       | 0       | 0       |
| Opt_3   | 0       | 0       | 0       | 0       | 0       | 1       |
| ...     | ...     | ...     | ...     | ...     | ...     | ...     |

All 120 options and all 6 assets are included, with all coefficients and references as retrieved.